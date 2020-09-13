import torch
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from os import path
from PIL import Image
from matplotlib import pyplot as plt


def show_images(images, titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def get_class_names(dataset_path='dataset'):
    dirname = path.dirname(__file__)
    dataset_path = path.join(dirname, dataset_path)
    dataset = datasets.ImageFolder(dataset_path)
    class_names = [item[0] for item in dataset.classes]
    return class_names


def init_data(dataset_path='dataset'):
    batch_size = 20
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dirname = path.dirname(__file__)
    dataset_path = path.join(dirname, dataset_path)
    dataset = datasets.ImageFolder(dataset_path, transform=data_transforms)

    dataset_len = len(dataset)
    train_len = int(0.6 * dataset_len)
    valid_len = int(0.2 * dataset_len)
    test_len = dataset_len - train_len - valid_len

    train_data, valid_data, test_data = random_split(dataset, [train_len, valid_len, test_len])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    class_names = [item[0] for item in dataset.classes]
    return loaders, class_names


def init_model():
    model = models.wide_resnet50_2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.fc.in_features
    alphabet_length = 52
    last_layer = torch.nn.Linear(n_inputs, alphabet_length)
    model.fc = last_layer
    model.fc.requires_grad = True
    return model


def train(n_epochs, loaders, model):
    valid_loss_min = np.Inf
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), "model.pt")
            valid_loss_min = valid_loss
        # return trained model
    return model


def test(loaders, model):
    test_loss = 0.
    correct = 0.
    total = 0.
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


def load_trained_model(model):
    dirname = path.dirname(__file__)
    model_path = path.join(dirname, 'model.pt')
    if path.exists(model_path):
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
        return
    print("Model needs to be trained first!")
    return


def predict(model, img, class_names):
    model.eval()
    pil_img = Image.fromarray(img.astype(float) * 255).convert('RGB')
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transformed_img = img_transforms(pil_img)
    transformed_img = torch.unsqueeze(transformed_img, 0)
    output = model(transformed_img)
    _, prediction = torch.max(output, 1)
    letter = class_names[prediction.item()]
    return letter


def main():
    loaders, class_names = init_data()
    model = init_model()
    # model = train(5, loaders, model)
    load_trained_model(model)
    test(loaders, model)


if __name__ == "__main__":
    main()
