import json
import os

target_file = 'sinewith_noise.ipynb'

if not os.path.exists(target_file):
    print(f"Error: {target_file} not found.")
    exit(1)

with open(target_file, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Define new content
new_model_code = [
    "class CurveFittingModel(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        # Improved Initialization to avoid local minima (Straight Line)\n",
    "        self.A1 = nn.Parameter(torch.randn(1, requires_grad=True))\n",
    "        self.A2 = nn.Parameter(torch.randn(1, requires_grad=True))\n",
    "        # Initialize frequencies to plausible range (10-20Hz) instead of ~0\n",
    "        self.f1 = nn.Parameter(torch.tensor([10.0], requires_grad=True))\n",
    "        self.f2 = nn.Parameter(torch.tensor([20.0], requires_grad=True))\n",
    "        self.theta1 = nn.Parameter(torch.rand(1, requires_grad=True) * 6.28)\n",
    "        self.theta2 = nn.Parameter(torch.rand(1, requires_grad=True) * 6.28)\n",
    "        self.b1 = nn.Parameter(torch.rand(1, requires_grad=True) * 10)\n",
    "        self.b2 = nn.Parameter(torch.rand(1, requires_grad=True) * 10)\n",
    "\n",
    "    def forward(self, t):\n",
    "        # โครงสร้างทางคณิตศาสตร์ของคลื่นผสมตามโจทย์\n",
    "        signal1 = self.A1 * torch.sin(self.f1 * t + self.theta1) + self.b1\n",
    "        signal2 = self.A2 * torch.cos(self.f2 * t + self.theta2) + self.b2\n",
    "        return signal1 + signal2\n",
    "\n",
    "# สร้าง object ของโมเดล\n",
    "model = CurveFittingModel()"
]

new_training_code = [
    "# 1. แปลงข้อมูล numpy ให้เป็น PyTorch tensors\n",
    "t_tensor = torch.from_numpy(t).float()\n",
    "y_tensor = torch.from_numpy(y).float()\n",
    "\n",
    "# Random Restart Strategy applied\n",
    "n_restarts = 10\n",
    "best_loss = float('inf')\n",
    "best_state = None\n",
    "\n",
    "print(\"Searching for best initialization...\")\n",
    "\n",
    "for attempt in range(n_restarts):\n",
    "    model = CurveFittingModel()\n",
    "    criterion = nn.MSELoss()\n",
    "    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)\n",
    "\n",
    "    # Quick training to test convergence\n",
    "    for i in range(1000):\n",
    "        optimizer.zero_grad()\n",
    "        y_pred = model(t_tensor)\n",
    "        loss = criterion(y_pred, y_tensor)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "    print(f\"Attempt {attempt+1}: Loss = {loss.item():.4f}\")\n",
    "    if loss.item() < best_loss:\n",
    "        best_loss = loss.item()\n",
    "        best_state = model.state_dict()\n",
    "        if best_loss < 20:\n",
    "            print(\"Found good start! Stopping search.\")\n",
    "            break\n",
    "\n",
    "print(f\"Best Loss found: {best_loss:.4f}\")\n",
    "model.load_state_dict(best_state)\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)\n",
    "\n",
    "epochs = 5000\n",
    "loss_history = []\n",
    "\n",
    "print(\"Starting fine-tuning...\")\n",
    "for i in range(epochs):\n",
    "    optimizer.zero_grad()\n",
    "    y_pred = model(t_tensor)\n",
    "    loss = criterion(y_pred, y_tensor)\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "\n",
    "    loss_history.append(loss.item())\n",
    "    if i % 500 == 0:\n",
    "        print(f\"Fine-tuning Epoch {i}: Loss = {loss.item():.4f}\")\n",
    "\n",
    "print(\"Training finished.\")"
]

model_updated = False
training_updated = False

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_str = "".join(source)
        
        if "class CurveFittingModel(nn.Module):" in source_str:
            print("Found Model Cell. Updating...")
            cell['source'] = new_model_code
            model_updated = True
            
        if "epochs = 5000" in source_str and "loss_history = []" in source_str and "Random Restart" not in source_str:
            print("Found Training Loop Cell. Updating...")
            cell['source'] = new_training_code
            training_updated = True

if model_updated and training_updated:
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print(f"Update incomplete. Model Updated: {model_updated}, Training Updated: {training_updated}")
