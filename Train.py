from torch.utils.data import DataLoader
from torchvision import transforms
from DataPrepare import InputData
from Network import DeepUNetTrainer


def train(args, device, datapath):
    # Get Data
    train_transform = transforms.ToTensor()
    validate_transform = transforms.ToTensor()
    train_data = InputData(transform=train_transform)
    val_data = InputData(datadir=datapath, mode='validate', transform=validate_transform)
    train_dataloader = DataLoader(dataset=train_data,batch_size=args.batch_size,shuffle=True)

    # Training
    # Todo: Implement Net
    Model = DeepUNetTrainer(args, train_dataloader, device)

    if args.train:
        iter_count = -1
        for epoch in range(1, args.epochsnum + 1) :
            iter_count = Model.train(iter_count)
            Model.validate(val_data, epoch, args.sample)
            Model.save(epoch)
            print("Epoch ", epoch, "complete")
    else:
        Model.validate(val_data, 1, args.sample)
