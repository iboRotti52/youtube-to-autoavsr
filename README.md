# 🎬 YouTube → Auto-AVSR Dataset Builder

YouTube videolarından Türkçe dudak okuma (**Auto-AVSR**) modeli eğitimi için yüksek kaliteli veri seti üreten otomatik pipeline.

> [!IMPORTANT]
> **Şu an SADECE `no_voiceover` (Dış sessiz) videolar toplanmaktadır!**
> - Kamera karşısında **konuşmacının yüzünün doğrudan ve net göründüğü** videoları ekliyoruz.
> - Arkadan seslendirme, dublaj, çevirmen sesi veya dış ses içeren videolar **eklenmemelidir**.
> - Projedeki tek aktif kaynak dosyamız: **`sources_no_voiceover.txt`**
> - Dosyaları elle düzenlemenize gerek yoktur; aşağıdaki komutlar listeyi, mükerrer kontrolünü ve GitHub senkronizasyonunu otomatik yönetir.

---

## ⚡ Günlük Ekip İş Akışı (Sadece 2 Adım)

Ekip üyelerinin günlük olarak yapacağı tek işlem bu iki adımdır:

### 1. Adım: Video Ekleme (`ytavsr add`)

Yeni videoları eklemek için terminalde komutu çalıştırmanız yeterlidir:

```bash
# Tek bir video eklemek:
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID"

# Veya kendi hazırladığınız bir metin dosyasındaki linkleri topluca eklemek:
ytavsr add linkler.txt
```

*(Bu komut otomatik olarak en güncel linkleri çeker (`git pull`), mükerrerleri ve önceden işlenmişleri eler, `sources_no_voiceover.txt` dosyasına ekler ve doğrudan GitHub'a yükler (`git push`)).*

---

### 2. Adım: Videoları Bulutta İşleme (`ytavsr modal`)

İşlemler (1080p indirme, Whisper transkripsiyon, RetinaFace yüz/dudak takibi ve ROI kesimi) **Modal GPU** üzerinde bulutta çalışır. Üretilen klipler doğrudan Hugging Face ortak depomuza (`avsr-tr-ekip/avsr-tr-dataset`) yüklenir; yerel diskiniz ve internet kotanız harcanmaz.

Ekipteki iş bölümü çakışmayı önlemek için **sabittir ve 3 parçaya bölünmüştür**:

| Ekip Üyesi | Sabit Shard | Çalıştırılacak Komut |
| :--- | :---: | :--- |
| **İbrahim Gözlükaya** | **Shard 0** | `ytavsr modal --shard 0` |
| **Damla Kemal** | **Shard 1** | `ytavsr modal --shard 1` |
| **İbrahim Billurcu** | **Shard 2** | `ytavsr modal --shard 2` |

> 💡 **Nasıl çalışır?** Listeye yeni videolar eklendikçe sistem daha önce işlenmiş videoları otomatik atlar ve kalan yeni videoları bu 3 shard'a paylaştırır. Her ekip üyesi sadece kendi shard komutunu çalıştırır.

---

## 🛠️ İlk Kez Kurulum (Sadece 1 Seferlik)

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

### 3. Hugging Face Girişi
1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) adresinden **Write** yetkili token oluşturun.
2. Terminalde giriş yapın:
   ```bash
   huggingface-cli login
   ```
*(Not: `avsr-tr-ekip` organizasyonuna davet edilmiş olmanız gerekir).*

### 4. Modal GPU Kurulumu
1. [modal.com](https://modal.com) adresinde GitHub ile ücretsiz hesap açın (aylık 30$ ücretsiz kredi tanımlanır).
2. Terminalde:
   ```bash
   pip install modal
   modal setup
   ```
*(Açılan tarayıcı penceresinden yetki vermeniz yeterlidir).*

---

## 📁 Üretilen Veri Yapısı

İşlenen klipler ortak Hugging Face deposunda (`avsr-tr-ekip/avsr-tr-dataset`) şu formatta saklanır:

```text
data/<uye_adi>/<video_id>/<segment_id>/
├── mouth.mp4          # Auto-AVSR için 96×96 ağız ROI videosu (25 fps)
├── transcript.txt     # Klibin konuşma metni (Whisper large-v3-turbo)
├── metadata.json      # Dudak senkronizasyonu ve kalite metrikleri
├── audio.wav          # Klip sesi (16 kHz mono)
└── source.mp4         # Ham video kesiti (debug amaçlı)
```

---

## 📖 Gelişmiş Seçenekler ve Ek Komutlar

Farklı parametreler, yerel (local) bilgisayarda çalıştırma, tek bir videoyu test etme, dış sesli (`voiceover`) video profili ve veri senkronizasyon komutları için:

👉 **[Gelişmiş Komutlar Rehberi (ADVANCED.md)](ADVANCED.md)** belgesine bakabilirsiniz.

---

## ⚖️ Lisans
Bu proje Apache-2.0 lisansı ile dağıtılmaktadır.
