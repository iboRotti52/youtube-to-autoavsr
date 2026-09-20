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

Ekip üyelerinin günlük olarak yapacağı işlemler bu 2 adımdan ibarettir:

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

### 2. Adım: Videoları İşleme

Ekipteki iş bölümü mükerrer çalışmayı önlemek için **sabittir ve 3 parçaya bölünmüştür**:

| Ekip Üyesi | Sabit Shard | Modal GPU Komutu (Önerilen) | Kendi Bilgisayarında (Lokal) |
| :--- | :---: | :--- | :--- |
| **İbrahim Gözlükaya** | **Shard 0** | `ytavsr modal --shard 0` | `ytavsr process-both-sources --shard 0` |
| **Damla Kemal** | **Shard 1** | `ytavsr modal --shard 1` | `ytavsr process-both-sources --shard 1` |
| **İbrahim Billurcu** | **Shard 2** | `ytavsr modal --shard 2` | `ytavsr process-both-sources --shard 2` |

#### Yöntem A: Modal Bulut GPU ile İşleme (Önerilen)
RetinaFace ve 1080p kesimleri Modal bulut GPU üzerinde çalışır. Üretilen klipler doğrudan Hugging Face ortak depomuza (`avsr-tr-ekip/avsr-tr-dataset`) yüklenir; yerel diskiniz ve internet kotanız harcanmaz.
```bash
ytavsr modal --shard 0    # (Shard numaranızı yazın: 0, 1 veya 2)
```
*(Modal container kendi bağımlılıklarını (ffmpeg, torch, RetinaFace/ibug) kendisi kurar; Modal kullanmak için local'de `setup-retinaface` yapmanıza **gerek yok**.)*

#### Yöntem B: Kendi Bilgisayarında (Lokal CPU/GPU) İşleme
Modal kullanmadan kendi bilgisayarınızda işlemek isterseniz. Lokal akış varsayılan olarak **MediaPipe** detector kullanır (Mac CPU'da ek kurulum gerekmez, `--config` vermenize gerek yok):
```bash
# 1. Kendi shard'ınızı yerelde işleyin:
ytavsr process-both-sources --shard 0  # (Shard numaranızı yazın: 0, 1 veya 2)

# 2. Üretilen kabul edilmiş klipleri Hugging Face deposuna yükleyin:
ytavsr push-data
```

Tek bir yerel video veya bir klasördeki tüm videoları işlemek için:
```bash
ytavsr process-local video.mp4          # tek dosya
ytavsr process-local ./videolar_klasoru/ # klasördeki .mp4/.mkv/.webm/.mov/.m4v dosyalarının hepsi (sıralı)
```

> 💡 **Nasıl çalışır?** Modal ve lokal akışta sıra aynıdır: önce iki kaynak listesinin tamamı deduplicate edilir, sonra tam liste shard'lara partition edilir, en son yalnızca atanmış shard içindeki processed kaynaklar atlanır. Böylece yeni videolar eklense veya başka shard tamamlanmış olsa da sahiplik kaymaz.

---

## 🛠️ Kendi Bilgisayarına Kurulum (Sadece 1 Seferlik)

Her ekip üyesi projeyi kendi bilgisayarına kurmak için aşağıdaki işletim sistemi adımlarını takip eder:

### 🍎 macOS & 🐧 Linux Kurulumu

**1. Sistem Araçlarını Yükleyin:**
```bash
# macOS:
brew install python@3.11 ffmpeg git git-lfs
git lfs install

# Linux (Ubuntu/Debian):
# sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg git git-lfs && git lfs install
```

**2. Projeyi İndirin ve Tek Komutla Kurun:**
```bash
git clone https://github.com/iboRotti52/youtube-to-autoavsr.git
cd youtube-to-autoavsr
./scripts/setup_once.sh
./scripts/install_terminal_command.sh
source ~/.zshrc    # Linux kullanıyorsanız: source ~/.bashrc
```
*(Bu sayede terminalde doğrudan `ytavsr` komutu tanımlanır).*
*(`setup_once.sh` lokal MediaPipe akışı için gerekenleri kurar: paket + Auto-AVSR + Whisper modeli. RetinaFace dahil **değildir**; GPU/Modal akışı için gerekirse ayrıca çalıştırın: `ytavsr setup-retinaface --config configs/retina_1080p.yaml`.)*

---

### 🪟 Windows Kurulumu

**1. Sistem Araçlarını Yükleyin (PowerShell Yönetici Olarak):**
```powershell
winget install Python.Python.3.11 Gyan.FFmpeg Git.Git GitHub.GitLFS
git lfs install
```
*(Kurulum bittikten sonra PowerShell'i kapatıp normal olarak yeniden açın).*

**2. Projeyi İndirin ve Otomatik Kurun:**
```cmd
git clone https://github.com/iboRotti52/youtube-to-autoavsr.git
cd youtube-to-autoavsr
scripts\setup_once.bat
```
*(Windows'ta komutları `scripts\run.bat add "URL"` veya sanal ortamı açarak `.venv\Scripts\activate` sonrasında `yt2avsr add "URL"` şeklinde çalıştırabilirsiniz).*

---

### 🔑 Hesap Girişleri (Tüm İşletim Sistemleri İçin Ortak)

**1. Hugging Face Girişi (Veri Yükleme Yetkisi):**
Ortak repoya veri gönderebilmek için Write yetkili Hugging Face token'ı gereklidir:
1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) adresinden **Write** yetkili token oluşturun.
2. Terminalde giriş yapın:
   ```bash
   huggingface-cli login
   ```
*(Not: `avsr-tr-ekip` organizasyonuna davet edilmiş olmanız gerekir).*

**2. Modal GPU Girişi (Bulutta Çalıştırmak İçin):**
1. [modal.com](https://modal.com) adresinde GitHub ile ücretsiz hesap açın (aylık 30$ ücretsiz GPU kredisi tanımlanır).
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
data/<contributor>/clips/<item>/<segment>/
├── mouth.mp4          # Auto-AVSR için 96×96 ağız ROI videosu (25 fps)
├── transcript.txt     # Klibin konuşma metni (Whisper large-v3-turbo)
├── metadata.json      # Dudak senkronizasyonu ve kalite metrikleri
├── audio.wav          # İsteğe bağlı; varsayılan HF upload'ında yoktur
└── source.mp4         # İsteğe bağlı debug dosyası; varsayılan HF upload'ında yoktur
```

Varsayılan `push-data` upload'ı `data/<contributor>/clips/<item>/<segment>` altında yalnızca eğitim için gereken dosyaları gönderir; `source.mp4` ve `audio.wav` varsayılan olarak gönderilmez.

---

## 📖 Gelişmiş Seçenekler ve Ek Komutlar

Farklı parametreler, tek bir videoyu test etme, dış sesli (`voiceover`) video profili ve veri senkronizasyon komutları için:

👉 **[Gelişmiş Komutlar Rehberi (ADVANCED.md)](ADVANCED.md)** belgesine bakabilirsiniz.

---

## ⚖️ Lisans
Bu proje Apache-2.0 lisansı ile dağıtılmaktadır.
