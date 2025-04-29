Diambil dari panduan installasi dlib pada repo [Github if4021](https://github.com/informatika-itera/if4021-handson/blob/main/2024/instalasi_dlib.md)

## **1. Instalasi Dependensi**

Sebelum menginstal `dlib`, Anda perlu menginstal beberapa dependensi:

- **CMake**: Diperlukan untuk membangun `dlib`.
- **Boost**: Beberapa versi `dlib` memerlukan Boost, terutama untuk performa yang lebih baik.
- **Python-dev** (untuk pengguna Linux): Header pengembangan Python diperlukan untuk mengkompilasi `dlib`.

---

## **2. Instalasi untuk Sistem Operasi yang Berbeda**

### **Windows**

#### **Menggunakan pip**

1. **Instal Python**: Pastikan Python sudah diinstal dan ditambahkan ke PATH.
2. **Instal CMake**: Unduh dan instal CMake dari [CMake website](https://cmake.org/download/), dan pastikan sudah ditambahkan ke PATH sistem.
3. **Instal Visual Studio Build Tools**:
   - Anda perlu menginstal Microsoft C++ Build Tools. Unduh dari [sini](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
   - Pilih "Desktop development with C++" workload, yang termasuk compiler MSVC.
4. **Instal dlib**:
   Buka command prompt (atau PowerShell) dan jalankan:

   ```bash
   pip install cmake
   pip install dlib
   ```

   Jika gagal, Anda bisa menggunakan pre-built wheels untuk `dlib` (lebih mudah):

   ```bash
   pip install dlib==19.22.0
   ```

#### **Menggunakan conda**

1. **Instal conda**: Jika Anda belum menginstal Anaconda atau Miniconda, dapatkan dari [Anaconda](https://www.anaconda.com/) atau [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **Buat environment baru**:

   ```bash
   conda create -n dlib_env python=3.8
   conda activate dlib_env
   ```

3. **Instal CMake**:

   ```bash
   conda install -c conda-forge cmake
   ```

4. **Instal dlib**:

   ```bash
   conda install -c conda-forge dlib
   ```

---

### **macOS**

#### **Menggunakan pip**

1. **Instal Python**: Anda bisa menginstal Python menggunakan `brew` atau mengunduhnya dari [Python site](https://www.python.org/).

   ```bash
   brew install python3
   ```

2. **Instal CMake**:

   ```bash
   brew install cmake
   ```

3. **Instal dlib**:

   ```bash
   pip install dlib
   ```

#### **Menggunakan conda**

1. **Instal conda**: Jika belum menginstal conda, dapatkan Anaconda atau Miniconda.
2. **Buat environment baru**:

   ```bash
   conda create -n dlib_env python=3.8
   conda activate dlib_env
   ```

3. **Instal CMake**:

   ```bash
   conda install -c conda-forge cmake
   ```

4. **Instal dlib**:

   ```bash
   conda install -c conda-forge dlib
   ```

---

### **Linux (Ubuntu)**

#### **Menggunakan pip**

1. **Instal dependensi**:

   ```bash
   sudo apt-get update
   sudo apt-get install build-essential cmake
   sudo apt-get install libgtk-3-dev
   sudo apt-get install libboost-all-dev
   sudo apt-get install python3-dev
   ```

2. **Instal dlib**:

   ```bash
   pip install dlib
   ```

   Jika instalasi gagal, coba gunakan pre-built binary wheel (lebih mudah):

   ```bash
   pip install dlib==19.22.0
   ```

#### **Menggunakan conda**

1. **Instal conda**: Instal Anaconda atau Miniconda jika belum terinstal.
2. **Buat environment baru**:

   ```bash
   conda create -n dlib_env python=3.8
   conda activate dlib_env
   ```

3. **Instal CMake**:

   ```bash
   conda install -c conda-forge cmake
   ```

4. **Instal dlib**:

   ```bash
   conda install -c conda-forge dlib
   ```

---

## **3. Verifikasi Instalasi**

Setelah menginstal `dlib`, Anda bisa memverifikasi instalasi dengan menjalankan skrip Python sederhana untuk mendeteksi wajah:

```python
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
img = cv2.imread('your_image.jpg')
faces = detector(img, 1)

print(f"Jumlah wajah yang terdeteksi: {len(faces)}")
```

Pastikan Anda telah menginstal OpenCV (`pip install opencv-python`) untuk menangani pembacaan gambar.

---

## **4. Catatan**

- Untuk dukungan **GPU**, Anda perlu mengompilasi `dlib` dari source dengan CUDA diaktifkan. Ini lebih kompleks, tetapi memberikan peningkatan performa jika Anda memiliki GPU NVIDIA.
- **Pre-built wheels**: Jika instalasi dari source gagal pada Windows atau Linux, Anda bisa sering menggunakan pre-built `dlib` binary, yang tersedia untuk versi Python tertentu. Menggunakan `pip install dlib==19.22.0` akan mencoba menarik versi pre-built jika tersedia.
- **Troubleshooting**: Jika Anda mengalami masalah dengan perpustakaan yang hilang (misalnya, kesalahan Boost atau CMake), periksa apakah variabel PATH Anda sudah diatur dengan benar dan semua dependensi sudah diinstal.

Dengan panduan ini, Anda seharusnya dapat menginstal `dlib` di berbagai sistem operasi dan package manager.
