from torch.utils.data import DataLoader
from dataset import UciDataset
from model import *
from utils import *
import constants as keys


# Function used for training a model with dropout
def train(dataset, dr, lr, bs, epochs, activate_pipeline=True):
    # Getting the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Choosing the configuration used for the selected dataset according to the Hernandez-Lobato experiment
    hidden_neurons = 100 if dataset in keys.LARGE_DATASETS else 50
    iterations = range(5) if dataset in keys.LARGE_DATASETS else range(20)

    # Iterating over the multiple splits
    for k in iterations:

        # Obtaining the dataset
        my_dataset = UciDataset(dataset, mode='training', split=k)
        my_dataloader = DataLoader(my_dataset, batch_size=bs, shuffle=True)
        path = obtain_training_folder(dataset, dr, lr, bs) if activate_pipeline else keys.TEMP_EXP_FOLDER
        path = os.path.join(path, 'cross_validation')
        if not os.path.exists(path):
            os.mkdir(path)

        # Setting up optimization parameters
        my_model = SimpleMLP(input_units=my_dataset.X_train.shape[1],
                             hidden_units=hidden_neurons, dropout_rate=dr).to(device).double()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=lr)
        loss_function = nn.MSELoss(reduction='mean')
        train_history, validation_history = [], []

        # Starting the training process
        for e in range(epochs):

            # Setting the model to train mode
            my_model.train()
            my_dataset.switch_mode('training')

            # Performing a first step of training
            for i_batch, sample_batched in enumerate(my_dataloader):

                # Getting the current batch
                X, y = sample_batched['x'].to(device), sample_batched['y'].to(device)
                if dataset in keys.UCI_DATASETS:
                    y = torch.unsqueeze(y, dim=1)
                pred = my_model(X)
                loss = loss_function(pred, y)

                # Performing Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluation sub-phase
            my_model.eval()

            # Getting global training and validation losses
            train_loss = compute_loss_on_dataloader(my_model, my_dataset, my_dataloader, device,
                                                    loss_function, mode='training')
            validation_loss = compute_loss_on_dataloader(my_model, my_dataset, my_dataloader, device,
                                                         loss_function, mode='validation')

            # Saving the loss values
            validation_history.append(validation_loss)
            train_history.append(train_loss)
            print("--- Epoch {}/{} ---\n Train Loss: {} / Validation Loss: {}".format(e, epochs, train_loss,
                                                                                      validation_loss))

        # --- Generating the loss image ---
        # Plotting train and validation histories
        plt.plot(train_history)
        plt.plot(validation_history)

        # Setting plot meta-data
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend(['Train', 'Validation'])

        # Saving and clearing the figure
        plt.savefig(os.path.join(path, 'loss_{}.png'.format(int(k))))
        plt.clf()

        # Saving the torch model
        torch.save(my_model.state_dict(), os.path.join(path, 'weights_{}.pt'.format(int(k))))
