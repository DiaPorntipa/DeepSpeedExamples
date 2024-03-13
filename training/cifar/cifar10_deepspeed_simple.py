import argparse

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from deepspeed.accelerator import get_accelerator


def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        # Local Rank
        # Dia: for testing.
        # Dia: local rank is a process id with respect to a given node. (https://medium.com/red-buffer/getting-started-with-pytorch-distributed-54ae933bb9f0)
        "--local_rank",
        type=int,
        default=-1,  # single-GPU training
        help="local rank passed from distributed launcher",  #? What is launcher?
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],  # Dia: https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407 
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="Datatype used for training",
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": 16,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],  
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(model_engine, testset, local_device, target_dtype, test_batch_size=4):
    """Test the network on the test data.

    Args:
        model_engine (deepspeed.runtime.engine.DeepSpeedEngine): the DeepSpeed engine.
        testset (torch.utils.data.Dataset): the test dataset.
        local_device (str): the local device name.
        target_dtype (torch.dtype): the target datatype for the test data.
        test_batch_size (int): the test batch size.

    """
    # The 10 classes for CIFAR10.
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Define the test dataloader.
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=0  # Dia: num_worker 0 means that the data will be loaded in the main process.
    )

    # For total accuracy.
    correct, total = 0, 0
    # For accuracy per class.
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    # Start testing.
    model_engine.eval()
    # temporarily disables gradient calculation during a specific block of code.
    with torch.no_grad(): 
        for data in testloader:
            images, labels = data
            if target_dtype != None:
                images = images.to(target_dtype)
            outputs = model_engine(images.to(local_device))
            _, predicted = torch.max(outputs.data, 1)  # Dia: select max values of each row. return tensor(values, induces)
            # Count the total accuracy.
            total += labels.size(0)
            correct += (predicted == labels.to(local_device)).sum().item()

            # Count the accuracy per class.
            batch_correct = (predicted == labels.to(local_device)).squeeze()  # numpy.squeeze(a, axis=None): Remove axes of length one from a.
            for i in range(test_batch_size):
                label = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label] += 1

    if model_engine.local_rank == 0:  #? Why only local_rank == 0, testing?
        print(
            f"Accuracy of the network on the {total} test images: {100 * correct / total : .0f} %"
        )

        # For all classes, print the accuracy.
        for i in range(10):
            print(
                f"Accuracy of {classes[i] : >5s} : {100 * class_correct[i] / class_total[i] : 2.0f} %"
            )


def main(args):
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()

    ########################################################################
    # Step1. Data Preparation.
    #
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].  # Dia: common preprocessing step
    #
    # Note:
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    ########################################################################
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # Dia: Subtract the mean value and divide the standard deviation.s
    )

    if torch.distributed.get_rank() != 0:  
        # Might be downloading cifar data, let rank 0 download first.  #? Why
        torch.distributed.barrier()  # a process is blocked by a barrier until all processes have encountered a barrier, upon which those barriers are lifted for all processes.

    # Load or download cifar data.
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    if torch.distributed.get_rank() == 0:  #? Why two times
        # Cifar data is downloaded, indicate other ranks can proceed.
        torch.distributed.barrier()

    ########################################################################
    # Step 2. Define the network with DeepSpeed.
    #
    # First, we define a Convolution Neural Network.
    # Then, we define the DeepSpeed configuration dictionary and use it to
    # initialize the DeepSpeed engine.
    ########################################################################
    net = Net(args)

    # Get list of parameters that require gradients.
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    # Initialize DeepSpeed to use the following features.
    #   1) Distributed model.
    #   2) Distributed data loader.
    #   3) DeepSpeed optimizer.
    ds_config = get_ds_config(args)
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=parameters,
        training_data=trainset,
        config=ds_config,
    )

    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    # Define the Classification Cross-Entropy loss function.
    criterion = nn.CrossEntropyLoss()

    ########################################################################
    # Step 3. Train the network.
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize. (DeepSpeed handles the distributed details for us!)
    ########################################################################

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # Get the inputs. ``data`` is a list of [inputs, labels].
            inputs, labels = data[0].to(local_device), data[1].to(local_device)  # Move the data to GPU.

            # Try to convert to target_dtype if needed.
            if target_dtype != None:
                inputs = inputs.to(target_dtype)

            outputs = model_engine(inputs)  #? Why sometimes loss = model_engine(batch)
            loss = criterion(outputs, labels)

            # runs backpropagation
            model_engine.backward(loss)
            # weight update
            model_engine.step()

            # Print statistics
            running_loss += loss.item()
            if local_rank == 0 and i % args.log_interval == (  #? why only when local_rank = 0
                args.log_interval - 1
            ):  # Print every log_interval mini-batches.
                print(
                    f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .3f}"
                )
                running_loss = 0.0
    print("Finished Training")

    ########################################################################
    # Step 4. Test the network on the test data.
    ########################################################################
    test(model_engine, testset, local_device, target_dtype)


if __name__ == "__main__":
    args = add_argument()
    main(args)
