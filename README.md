# 🎬 YouTube → Auto-AVSR Dataset Builder (v0.7)

YouTube videolarından Türkçe dudak okuma (**Auto-AVSR**) modeli eğitimi için yüksek kaliteli veri seti üreten uçtan uca otomatik pipeline.

### Pipeline Adımları
1. **İndirme (1080p):** YouTube videolarını en yüksek kalitede (1080p, 25 fps) indirir.
2. **Transkripsiyon (Whisper large-v3-turbo):** Videoyu CUDA/CPU üzerinde Whisper ile yüksek doğrulukla yazıya döker ve kelime zamanlamalarını çıkarır.
3. **Segmentasyon:** Cümle/duraklama sınırlarına ve sahne değişimlerine göre 2-16 saniyelik temiz parçalara böler.
4. **Yüz & Dudak ROI Kesimi (96×96):**
   - **Modal GPU:** Resmi Auto-AVSR **RetinaFace** + FAN yüz takipçisi (`mouth.mp4`).
   - **Yerel Mac/CPU:** Hızlı **MediaPipe** yüz takipçisi.
5. **Kalite & Dudak Senkronizasyonu (Lip-Sync):** Dış ses/dublaj tespiti, ağız açıklığı, hareket ve benzerlik filtreleri uygular; veriyi `accepted`, `review`, `rejected` olarak gruplar.
6. **Hugging Face Senkronizasyonu:** Kabul edilen klipleri doğrudan ortak bulut deposuna (`iboRotti/avsr-tr-dataset`) yükler.

---

## ⚡ Hızlı Başlangıç (Ekip İş Akışı)

Projede ekip olarak çalışırken günlük iş akışı 3 basit adımdan oluşur:

### 1. Link Ekleme (`ytavsr add`)
Linkleri dosyalara elle eklemek yerine her zaman **`ytavsr add`** komutunu kullanın. Bu komut otomatik olarak:
- Ekip arkadaşlarının eklediği son linkleri çeker (`git pull`).
- Mükerrer (daha önce işlenmiş veya listede olan) linkleri filtreler.
- Dosyaya ekleyip GitHub'a otomatik yükler (`git push`).

```bash
# 1. Normal konuşma videosu (ekranda konuşan var, dış ses/dublaj yok):
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID"

# 2. Birden fazla videoyu aynı anda ekleme:
ytavsr add "https://youtu.be/VID1" "https://youtu.be/VID2"

# 3. Dış ses / dublaj / belgesel / anlatıcı içeren video:
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID" --voiceover

# 4. Bir metin dosyasındaki tüm linkleri topluca ekleme:
ytavsr add linkler.txt
```

---

### 2. Videoları Modal GPU ile Bulutta İşleme (Önerilen)
RetinaFace ve 1080p işlemi yerel bilgisayarlarda (özellikle Mac CPU) çok yavaş kalabileceğinden, her ekip üyesi kendi kişisel ve ücretsiz **Modal** (modal.com) GPU hesabında çalıştırır. Videolar buluttan doğrudan Hugging Face'e aktarılır (ev/ofis internetinizin yükleme kotası ve yerel diskiniz harcanmaz).

#### Sabit Ekip Shard Dağılımı (Asla Değişmez):
Ekipte mükerrer çalışmayı önlemek için iş bölümü sabittir:

| Ekip Üyesi | Sabit Shard | Modal GPU Komutu | Yerel (Local) Komut |
| :--- | :---: | :--- | :--- |
| **İbrahim Gözlükaya** | **Shard 0** (`0/3`) | `ytavsr modal --shard 0` | `ytavsr --shard 0` |
| **Damla Kemal** | **Shard 1** (`1/3`) | `ytavsr modal --shard 1` | `ytavsr --shard 1` |
| **İbrahim Billurcu** | **Shard 2** (`2/3`) | `ytavsr modal --shard 2` | `ytavsr --shard 2` |

> 💡 **Not:** Listeye yeni videolar eklendikçe sistem işlenenleri otomatik atlar ve yeni videoları sırasıyla bu 3 shard'a paylaştırır. Herkes her zaman sadece kendi shard komutunu çalıştırır.

```bash
# Kendi shard'ınızı Modal bulut GPU'sunda çalıştırmak:
ytavsr modal --shard 0     # İbrahim Gözlükaya
ytavsr modal --shard 1     # Damla Kemal
ytavsr modal --shard 2     # İbrahim Billurcu

# Tek bir videoyu Modal GPU'sunda test etmek:
ytavsr modal --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Dış sesli tek bir videoyu Modal'da test etmek:
ytavsr modal --url "https://www.youtube.com/watch?v=VIDEO_ID" --voiceover
```

---

### 3. Yerel (Local) Çalıştırma (Alternatif)
Modal kullanmadan kendi bilgisayarınızda (CPU veya yerel GPU) çalıştırmak isterseniz:

```bash
# Kendi shard'ınızı yerelde işlemek:
ytavsr --shard 0     # İbrahim Gözlükaya
ytavsr --shard 1     # Damla Kemal
ytavsr --shard 2     # İbrahim Billurcu

# Tek başına tüm listeyi işlemek:
ytavsr
```

Yerelde üretilen verileri Hugging Face deposuna yüklemek için:
```bash
ytavsr push-data
```

---

## 🛠️ İlk Kurulum Rehberi (Sadece 1 Kez)

### 1. Sistem Araçları (macOS)
```bash
brew install python@3.11 ffmpeg git git-lfs
git lfs install
```

### 2. Proje Kurulumu
```bash
git clone https://github.com/iboRotti52/youtube-to-autoavsr.git
cd youtube-to-autoavsr
./scripts/setup_once.sh
./scripts/install_terminal_command.sh
source ~/.zshrc
```
*(Bu sayede terminalde doğrudan `ytavsr` komutu kullanılabilir).*

### 3. Hugging Face Girişi (Ön Koşul)
Modal veya yerelde üretilen kliplerin ortak depoya aktarılabilmesi için Write yetkili HF token'ı gereklidir:
1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) adresinden **Write** yetkili token oluşturun.
2. Terminalde çalıştırıp token'ı yapıştırın:
   ```bash
   huggingface-cli login
   ```

### 4. Modal Kurulumu (Bulut GPU İçin)
1. [modal.com](https://modal.com) adresinde GitHub ile ücretsiz hesap açın (aylık 30$ ücretsiz GPU kredisi).
2. Terminalde:
   ```bash
   pip install modal
   modal setup
   ```
*(Açılan tarayıcı sekmesinden yetki vermeniz yeterlidir).*

---

## 📂 Kaynak Dosyaları ve Video Profilleri

| Profil | Dosya | Ekleme | Açıklama |
| :--- | :--- | :--- | :--- |
| `no_voiceover` | `sources_no_voiceover.txt` | `ytavsr add "URL"` | Ekranda konuşan kişi var, dış ses yok. Lip-sync toleranslıdır (Varsayılan). |
| `voiceover` | `sources_voiceover.txt` | `ytavsr add "URL" --voiceover` | Dublaj, anlatıcı, dış ses içerebilir. Ses varken ağız oynamıyorsa klip elenir. |
| Takip | `processed_sources.txt` | Otomatik | Ekipçe daha önce işlenmiş videoların listesi. Tekrar indirilip işlenmez. |

---

## 📁 Çıktı Yapısı

İşlenen klipler `data/clips/` altında, manifestler ise `data/manifests/` altında oluşur:

```text
data/
├── clips/<video_id>/<segment_id>/
│   ├── mouth.mp4          # Auto-AVSR için 96×96 ağız ROI videosu (25 fps)
│   ├── transcript.txt     # Klibin konuşma metni (Whisper)
│   ├── metadata.json      # Dudak senkronizasyonu ve kalite metrikleri
│   ├── audio.wav          # Klip sesi (16 kHz mono)
│   └── source.mp4         # Ham video kesiti (debug)
└── manifests/
    ├── accepted.csv       # Eğitim için onaylanan temiz klipler
    ├── review.csv         # İnceleme gerektiren şüpheli klipler
    └── rejected.csv       # Standartları sağlamayan elenmiş klipler
```

---

## 📋 Hızlı Komut Referansı

| Komut | Açıklama |
| :--- | :--- |
| `ytavsr add "URL"` | Yeni video linki ekler, mükerrerleri önler, GitHub'a pushlar |
| `ytavsr add "URL" --voiceover` | Dış sesli/dublajlı video ekler |
| `ytavsr modal --shard 0` | İbrahim Gözlükaya için Modal GPU'da işleme (Buluttan HF'ye) |
| `ytavsr modal --shard 1` | Damla Kemal için Modal GPU'da işleme (Buluttan HF'ye) |
| `ytavsr modal --shard 2` | İbrahim Billurcu için Modal GPU'da işleme (Buluttan HF'ye) |
| `ytavsr modal --url "URL"` | Tek bir videoyu Modal GPU'da işler |
| `ytavsr --shard 0` | İbrahim Gözlükaya için yerel bilgisayarda işleme |
| `ytavsr --shard 1` | Damla Kemal için yerel bilgisayarda işleme |
| `ytavsr --shard 2` | İbrahim Billurcu için yerel bilgisayarda işleme |
| `ytavsr` | Yerel bilgisayarda bekleyen tüm videoları işler |
| `ytavsr sync-processed` | Hugging Face'teki son işlenmiş videoları yerel listeye çeker |
| `ytavsr push-data` | Yereldeki kabul edilen klipleri Hugging Face'e yükler |
| `ytavsr pull-data` | Eğitim makinesinde Hugging Face'teki tüm ekip verisini indirir |
| `ytavsr dedup-sources` | Kaynak dosyalarındaki mükerrer satırları temizler |

---

## ⚖️ Lisans
Bu proje Apache-2.0 lisansı ile dağıtılmaktadır.
