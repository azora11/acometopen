Menentukan Jalur Terpendek Menggunakan Algoritma Ant Colony 
Optimization 

Metodologi Eksperimen 
Eksperimen sederhana yang dilakukan ini bertujuan untuk memahami konsep dasar dan cara 
kerja ACO (Ant Colony Optimization) dalam menemukan jalur optimal berdasarkan feromon 
yang digunakan oleh semut virtual. 

1. Data dan Persiapan - 
Dataset yang digunakan: desa_baru.csv dengan jarak antar 11 desa.
Matriks jarak antar desa disiapkan dalam format CSV dan dibaca menggunakan 
pandas. 
Model ACO (Ant Colony Optimization) diimplementasikan menggunakan Python 
dengan parameter berikut: 
a. Jumlah semut: 10 
b. Iterasi: 100 
c. Evaporasi feromon: 0.95 
d. Faktor eksplorasi (α): 1 
e. Faktor eksploitasi (β): 2

2. Proses Eksperimen
Semut virtual memulai pencarian jalur dari titik acak. 
Setiap semut memilih jalur berdasarkan intensitas feromon dan jarak. 
Feromon diperbarui setelah setiap iterasi, dengan jalur yang lebih pendek 
mendapatkan feromon lebih banyak. 
Algoritma mencari jalur terpendek di antara desa-desa berdasarkan pembaruan 
feromon dan eksplorasi jalur baru. 
