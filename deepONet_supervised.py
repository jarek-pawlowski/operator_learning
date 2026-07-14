import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import utils

n_sensors = 50
n_query = 100

sensor_x = torch.linspace(0.0, 1.0, n_sensors)  # sensor points = positions of f-values (f representation) [M]
query_x = torch.linspace(0.0, 1.0, n_query)     # query points = positions where we want to predict u (u representation) [Q]

f_train, x_train, u_train = utils.generate_functions(
    n_functions=2000,
    sensor_x=sensor_x,
    query_x=query_x,
    n_modes=5,
)

train_dataset = TensorDataset(f_train, x_train, u_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)


# ---------------------------------------------------------
# Training DeepONet
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.DeepONet(
    n_sensors=n_sensors,
    latent_dim=64,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
loss_history = []
for epoch in range(500):
    model.train()
    epoch_loss = 0.0

    for f_batch, x_batch, u_batch in train_loader:
        f_batch = f_batch.to(device)
        x_batch = x_batch.to(device)
        u_batch = u_batch.to(device)

        optimizer.zero_grad()

        prediction = model(f_batch, x_batch)
        loss = loss_fn(prediction, u_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 10 == 0:
        average_loss = epoch_loss / len(train_loader)
        loss_history.append(average_loss)
        print(f"epoch={epoch:4d}, loss={average_loss:.4e}")


# ---------------------------------------------------------
# Test using new function
# ---------------------------------------------------------

model.eval()

f_test, x_test, u_exact = utils.generate_functions(
    n_functions=1,
    sensor_x=sensor_x,
    query_x=query_x,
    n_modes=5,
)
with torch.no_grad():
    u_pred = model(
        f_test.to(device),
        x_test.to(device),
    ).cpu()
relative_error = (
    torch.linalg.norm(u_pred - u_exact) / torch.linalg.norm(u_exact)
)
print("Relative L2 error:", relative_error.item())
utils.plot_loss(loss_history, no_epochs_per_tick=10)
utils.plot_predictions(x_test, u_exact, u_pred, [sensor_x, f_test])