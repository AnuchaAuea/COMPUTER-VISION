# คู่มือการทำโจทย์ PyTorch Curve Fitting (ฉบับปรับปรุง)

## สารบัญ
1.  [การเตรียมข้อมูล (Data Preparation)](#1-การเตรียมข้อมูล-data-preparation)
2.  [สร้างโมเดล (Model Definition)](#2-สร้างโมเดล-model-definition) **(อัปเดตวิธีแก้ปัญหากราฟเส้นตรง)**
3.  [Loss Function และ Optimizer](#3-loss-function-และ-optimizer)
4.  [การเทรนโมเดล (Training Loop)](#4-การเทรนโมเดล-training-loop)
5.  [ตรวจสอบผลลัพธ์ (Result Verification)](#5-ตรวจสอบผลลัพธ์-result-verification)

---

## ปัญหา "กราฟเป็นเส้นตรง" (Local Minima Issue)

**ถาม:** ทำไมรันแล้วได้กราฟเส้นตรง ไม่เป็นคลื่นไซน์?
**ตอบ:** ปัญหานี้เกิดจาก **การสุ่มค่าเริ่มต้น (Initialization)** ครับ

โจทย์ข้อนี้กราฟมีความถี่สูง (f1 ~ 10-15, f2 ~ 20-30) แต่คำสั่ง `torch.randn(1)` จะสุ่มค่าออกมาใกล้ๆ 0 เสมอ เมื่อโมเดลเริ่มต้นด้วยความถี่ 0 (เส้นตรง) มันจะพยายามปรับค่าทีละนิด แต่หาคลื่นไม่เจอเพราะความถี่จริงอยู่ไกลเกินไป ทำให้โมเดล "ติดหล่ม" (Local Minimum) และเรียนรู้ได้แค่ค่าเฉลี่ย (Bias) เท่านั้น

**วิธีแก้:** เราต้อง "ใบ้" ค่าเริ่มต้นให้โมเดลหน่อยครับ โดยเปลี่ยนจากการสุ่มมั่วๆ เป็นการกำหนดค่าเริ่มต้นให้อยู่ในช่วงที่สมเหตุสมผล เช่น f เริ่มต้นที่ 10-20 แทนที่จะเป็น 0

---

## 2. สร้างโมเดล (Model Definition) - ฉบับปรับปรุง

ให้แทนที่ Class `CurveFittingModel` เดิมด้วยโค้ดชุดนี้ครับ (มีการปรับปรุงส่วน `__init__`):

```python
class CurveFittingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # --- ปรับปรุงการกำหนดค่าเริ่มต้น (Initialization) ---
        # แทนที่จะสุ่มมั่วๆ เราจะสุ่มในช่วงที่กว้างขึ้นหรือกำหนดค่าที่ใกล้เคียงความจริง
        
        # A (Amplitude): สุ่มปกติได้ ค่าน้อยๆ ไม่เป็นไร
        self.A1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.A2 = nn.Parameter(torch.randn(1, requires_grad=True))

        # f (Frequency): สำคัญมาก! ต้องเริ่มด้วยค่าที่สูงหน่อย (เช่น 10-20)
        # ถ้าเริ่มที่ 0 โมเดลจะหาคลื่นไม่เจอ
        self.f1 = nn.Parameter(torch.tensor([10.0], requires_grad=True)) # เดาว่า ~10 Hz
        self.f2 = nn.Parameter(torch.tensor([20.0], requires_grad=True)) # เดาว่า ~20 Hz

        # theta (Phase): สุ่มช่วง 0 ถึง 2pi
        self.theta1 = nn.Parameter(torch.rand(1, requires_grad=True) * 6.28) 
        self.theta2 = nn.Parameter(torch.rand(1, requires_grad=True) * 6.28)

        # b (Bias): ค่าเฉลี่ยอยู่นอกศูนย์ ควรสุ่มค่าบวก (0-10)
        self.b1 = nn.Parameter(torch.rand(1, requires_grad=True) * 10) 
        self.b2 = nn.Parameter(torch.rand(1, requires_grad=True) * 10)

    def forward(self, t):
        # สมการเหมือนเดิม
        signal1 = self.A1 * torch.sin(self.f1 * t + self.theta1) + self.b1
        signal2 = self.A2 * torch.cos(self.f2 * t + self.theta2) + self.b2
        return signal1 + signal2

# สร้าง object ของโมเดล
model = CurveFittingModel()
```

---

## 3. Loss Function และ Optimizer

เหมือนเดิมครับ แต่แนะนำให้ลองรันซ้ำหลายๆ รอบถ้าผลยังไม่ดี

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # Learning rate 0.1 กำลังดี
```

---

## 4. การเทรนโมเดล (Training Loop)

ส่วนนี้ใช้โค้ดเดิมได้เลยครับ:

```python
# แปลงข้อมูลเป็น Tensor (ถ้ายังไม่ได้ทำ)
t_tensor = torch.from_numpy(t).float()
y_tensor = torch.from_numpy(y).float()

epochs = 5000 # เทรน 5000 รอบ
loss_history = []

print("Starting training...")
for i in range(epochs):
    optimizer.zero_grad()           # 1. ล้าง Gradient เก่า
    y_pred = model(t_tensor)        # 2. ทำนายผล
    loss = criterion(y_pred, y_tensor) # 3. เทียบความต่าง (Loss)
    loss.backward()                 # 4. คำนวณ Gradient (หาทิศทางปรับ)
    optimizer.step()                # 5. ปรับค่าพารามิเตอร์

    loss_history.append(loss.item())
    if i % 500 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

print("Training finished.")
```

---

## 5. ตรวจสอบผลลัพธ์ (Result Verification)

โค้ดส่วนแสดงผลเหมือนเดิมครับ หลังจากแก้ `__init__` และรันใหม่ กราฟควรจะเป็นคลื่นไซน์ที่ทับกับข้อมูลจริงได้สวยงามครับ

```python
with torch.no_grad():
   y_predicted = model(t_tensor).numpy()

plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Ground Truth', alpha=0.5)
plt.plot(t, y_predicted, 'r--', label='Model Prediction', linewidth=2)
plt.legend()
plt.title("Curve Fitting Result (Improved)")
plt.show()

# ค่า Ground Truth ดูจากด้านบนสุดของ Notebook

---

## 6. โบนัส: วิธีแก้ปัญหาแบบขั้นเทพ (Random Restarts)

ถ้าลองปรับค่าเริ่มต้นแล้วยังไม่หาย หรือขี้เกียจมานั่งสุ่มค่าเอง ให้ใช้เทคนิค **"Random Restarts"** ครับ คือเขียนโค้ดให้มันลองรัน Loop สั้นๆ หลายๆ รอบ ถ้าค่า Loss ยังสูง ให้ "รีเซ็ตค่าพารามิเตอร์" แล้วเริ่มใหม่ จนกว่าจะเจอจุดเริ่มต้นที่ดีครับ

ก๊อปปี้โค้ดชุดนี้ไปรันแทน Training Loop เดิมได้เลย รับรองกราฟสวยแน่นอน!

```python
# จำนวนรอบที่จะลองสุ่มใหม่ (Try count)
n_restarts = 10
best_loss = float('inf')
best_state = None

print("Searching for best initialization...")

for attempt in range(n_restarts):
    # 1. สร้างโมเดลใหม่ทุกครั้ง (Re-initialize parameters)
    model = CurveFittingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # LR สูงหน่อยช่วงแรก

    # 2. ลองเทรนสั้นๆ 1000 Epochs ดูแนวโน้ม
    for i in range(1000):
        optimizer.zero_grad()
        y_pred = model(t_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
    
    current_loss = loss.item()
    print(f"Attempt {attempt+1}: Loss = {current_loss:.4f}")
    
    # 3. ถ้า Loss ต่ำกว่าเดิม ให้เก็บสถานะนี้ไว้ (Save best model)
    if current_loss < best_loss:
        best_loss = current_loss
        best_state = model.state_dict()
        if best_loss < 20: # ถ้า Loss ต่ำกว่า 20 ถือว่าใช้ได้แล้ว หยุดหาเลย
            print("Found good start! Stopping search.")
            break

print(f"\nBest Loss found: {best_loss:.4f}")
print("Loading best parameters and continuing training...")

# 4. โหลดค่าที่ดีที่สุดกลับมา แล้วเทรนต่อยาวๆ เพื่อความแม่นยำ (Fine-tuning)
model.load_state_dict(best_state)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # ลด LR ลงเพื่อความละเอียด

for i in range(5000): # เทรนต่ออีก 5000 รอบ
    optimizer.zero_grad()
    y_pred = model(t_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f"Fine-tuning Epoch {i}: Loss = {loss.item():.4f}")

print("Final Loss:", loss.item())
```

