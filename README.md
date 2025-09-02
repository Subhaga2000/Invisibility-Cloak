# 🧙‍♂️ Invisibility Cloak using Python & OpenCV  

A fun mini-project that recreates the famous Harry Potter **Invisibility Cloak** effect using Python and OpenCV. This project teaches you concepts like **image segmentation, color detection, and real-time video processing** while having fun! 🚀  

---

## ✨ How it Works
1. **Capture the Background** – Records the background before you appear with the cloak.  
2. **Color Detection** – Detects the cloak color (e.g., red) using the HSV color space.  
3. **Mask Refinement** – Cleans up the detected region with morphological operations.  
4. **Region Replacement** – Replaces the cloak region with the background pixels.  
5. **Display** – Shows the final invisibility effect live using your webcam.  

---

## 🛠 Tech Stack  
- **Python** 🐍 – Core programming  
- **OpenCV** 📹 – Real-time video processing  
- **NumPy** 🔢 – Array operations  
- **VS Code** 💻 – Development  

---

## 📦 Installation & Setup  

### 1. Clone the Repository  

git clone https://github.com/Subhaga2000/Invisibility-Cloak.git
cd Invisibility-Cloak

### 2. Install Dependencies

pip install -r requirements.txt


### 3. Run the Project

python cloak.py