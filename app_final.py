import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import streamlit as st
import graphviz

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

search_space = {
    0: ('conv3x3', 16),
    1: ('conv3x3', 32),
    2: ('conv5x5', 16),
    3: ('conv5x5', 32),
    4: ('maxpool', "k=2"),
    5: ('batchnorm', "auto"),
    6: ('dropout', 0.25),
    7: ('dropout', 0.5)
}

space_size = len(search_space)

def one_hot(idx, space_size):
    vec = torch.zeros(1, 1, space_size)
    vec[0, 0, idx] = 1
    return vec


class ControllerRNN(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super(ControllerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(space_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, space_size)

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        logits = self.linear(output.squeeze(0))
        return logits, hidden

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))


class ChildModel(nn.Module):
    def __init__(self, architecture):
        super(ChildModel, self).__init__()
        layers = []
        in_channels = 1
        for step in architecture:
            layer_type, param = search_space[step]
            if layer_type == 'conv3x3':
                layers.append(nn.Conv2d(in_channels, param, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                in_channels = param
            elif layer_type == 'conv5x5':
                layers.append(nn.Conv2d(in_channels, param, kernel_size=5, padding=2))
                layers.append(nn.ReLU())
                in_channels = param
            elif layer_type == 'maxpool':
                layers.append(nn.MaxPool2d(kernel_size=2))
            elif layer_type == 'batchnorm':
                layers.append(nn.BatchNorm2d(in_channels))
            elif layer_type == 'dropout':
                layers.append(nn.Dropout(param))

        layers.append(nn.AdaptiveAvgPool2d((3, 3)))  # unify size
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels * 3 * 3, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_child(architecture, train_loader, val_loader, lr=0.01):
    model = ChildModel(architecture)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx > 100:  # train only first ~6400 samples for speed
            break

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
    val_acc = correct / total
    return val_acc


def architecture_to_graph(architecture, search_space):
    dot = graphviz.Digraph(comment="Best Architecture", format="png")
    dot.attr(rankdir='LR')
    prev = "Input"
    dot.node(prev, shape='box')
    for i, step in enumerate(architecture):
        label = f"{search_space[step][0]} ({search_space[step][1]})" if search_space[step][1] else search_space[step][0]
        curr = f"Layer {i+1}"
        dot.node(curr, label)
        dot.edge(prev, curr)
        prev = curr
    dot.node("Output", shape='box')
    dot.edge(prev, "Output")
    return dot


st.title("NAS with Controller RNN")

num_layers_to_choose = st.slider("Number of Layers to Choose", min_value=1, max_value=10, value=6)
episodes = st.slider("Number of Episodes", 1, 20, 5)
batch_size = st.selectbox("Batch Size", [16, 32, 64], index=2)
learning_rate = st.number_input("Learning Rate (Child Model)", value=0.01, format="%.5f")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

if st.button("🚀 Start Training"):
    st.write("Training started...")
    reward_list = []
    episode_archs = []
    progress = st.progress(0)
    controller = ControllerRNN()
    optimizer = optim.Adam(controller.parameters(), lr=0.01)

    for episode in range(episodes):
        hidden = controller.init_hidden()
        inputs = one_hot(0, space_size)
        log_probs = []
        architecture = []

        for _ in range(num_layers_to_choose):
            logits, hidden = controller(inputs, hidden)
            probs = torch.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
            architecture.append(action.item())
            inputs = one_hot(action.item(), space_size)

        arch_desc = [search_space[x] for x in architecture]
        st.write(f"Episode {episode+1}/{episodes} Architecture: {arch_desc}")

        reward = train_child(architecture, train_loader, val_loader, lr=learning_rate)
        reward_list.append(reward)
        episode_archs.append(architecture)

        st.write(f"Episode {episode+1}/{episodes} Validation Accuracy: {reward:.4f}")

        loss = -torch.stack(log_probs).sum() * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.progress((episode + 1) / episodes)

    st.subheader("Reward Progress")
    st.line_chart(reward_list)
    if episode_archs:
        best_idx = max(range(len(reward_list)), key=lambda i: reward_list[i])
        best_arch = episode_archs[best_idx]
        st.subheader(f"🏆 Best Architecture (Episode {best_idx+1})")
        st.json([search_space[x] for x in best_arch])
        dot = architecture_to_graph(best_arch, search_space)
        st.graphviz_chart(dot.source)
else:
    st.info("Adjust the parameters above and press **Start Training** to begin.")
