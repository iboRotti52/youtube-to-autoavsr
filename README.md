# YouTube -> Auto-AVSR Dataset Builder v0.7

Bu proje, izin verilen YouTube videolarindan Auto-AVSR tarzinda dudak okuma
egitim verisi uretir.

Kisaca yaptigi is:

1. YouTube videosunu indirir.
2. Turkce altyazi varsa onu kullanir.
3. Altyazi yoksa Whisper ile Turkce transcript uretir.
4. Videoyu cumle/segment parcalarina boler.
5. Resmi Auto-AVSR preprocessing ile 96x96 agiz videosu (`mouth.mp4`) uretir.
6. Kotu veya supheli klipleri `accepted`, `review`, `rejected` olarak ayirir.

> Bu pipeline tek konusmacili, ekranda yuzu gorunen videolar icin tasarlanmistir.
> Panel, roportaj, coklu yuz veya ekranda konusmayan dis ses agirlikli videolar
> icin uygun degildir.

## Hangi Dosyaya Link Yazacagim?

Repo icinde iki kaynak dosyasi var:

```text
sources_no_voiceover.txt
sources_voiceover.txt
```

Video turune gore linki bu dosyalardan birine yaz:

| Dosya | Ne zaman kullanilir? |
| --- | --- |
| `sources_no_voiceover.txt` | Ekrandaki kisi konusuyor, dis ses/dublaj yok. |
| `sources_voiceover.txt` | Anlatici, dis ses, dublaj veya ekrandaki agizla her zaman eslesmeyen ses olabilir. |

Her satira bir YouTube linki yaz:

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://www.youtube.com/playlist?list=PLAYLIST_ID
```

Yorum eklemek icin satirin basina `#` koyabilirsin:

```text
# Tek konusmacili egitim videolari
https://www.youtube.com/watch?v=VIDEO_ID
```

## En Kisa Kullanim

Kurulumdan sonra sadece sunu calistir:

```bash
ytavsr
```

Bu komut:

- `sources_no_voiceover.txt` dosyasindaki linkleri `no_voiceover` profiliyle,
- `sources_voiceover.txt` dosyasindaki linkleri `voiceover` profiliyle

isler. Bos dosyalari atlar.

## Guclu Bilgisayarlar Icin RetinaFace

Varsayilan ayar 1080p + MediaPipe'tir. Mac CPU gibi yerel makinelerde hizli
veri uretmek icin uygundur. Guclu GPU'lu bilgisayarlarda veya Colab'da resmi
Auto-AVSR akisine daha yakin RetinaFace crop kullanabilirsin:

```bash
ytavsr process-both-sources --config configs/retinaface.yaml
```

Bu ayar:

- Varsayilan 1080p indirme/normalizasyon ayarlarini korur.
- Auto-AVSR crop dedektorunu MediaPipe yerine RetinaFace yapar.
- Auto-AVSR agiz ciktisini yine 96x96 `mouth.mp4` olarak uretir.

RetinaFace daha fazla RAM ve islem suresi kullanir. Mac CPU'da varsayilan
`ytavsr` komutu yani MediaPipe onerilir.

## Bastan Kurulum: macOS / Linux

### 1. Projeyi indir

GitHub'dan indiriyorsan:

```bash
git clone REPO_URL
cd youtube-to-autoavsr-v0.7-
```

ZIP olarak indirdiysen klasoru ac ve terminalde proje klasorune gir:

```bash
cd /path/to/youtube-to-autoavsr-v0.7-
```

### 2. Sistem araclarini kur

macOS:

```bash
brew install python@3.11 ffmpeg git git-lfs
git lfs install
```

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv ffmpeg git git-lfs
git lfs install
```

### 3. Proje kurulumunu calistir

Proje klasorundeyken:

```bash
./scripts/setup_once.sh
```

Bu komut sunlari hazirlar:

- `.venv` sanal Python ortamini olusturur.
- Python paketlerini kurar.
- Resmi Auto-AVSR reposunu `external/auto_avsr` altina indirir.
- RetinaFace / ibug yuz tespit paketlerini kurar.
- Whisper modelini ilk calistirmadan once indirir.

### 4. Terminal kisayolunu ekle

```bash
./scripts/install_terminal_command.sh
source ~/.zshrc
```

Bundan sonra proje klasorune girmeden terminalden `ytavsr` yazabilirsin.

Kurulumu tekrar hazirlamak veya guncellemek gerekirse:

```bash
ytavsr-setup
```

### 5. Linkleri ekle ve calistir

Linkleri `sources_no_voiceover.txt` veya `sources_voiceover.txt` dosyasina yaz.
Sonra:

```bash
ytavsr
```

## Bastan Kurulum: Windows

Windows'ta Python 3.11 kullan. `mediapipe 0.10.x`, Python 3.12+ ile sorun
cikarabilir.

### 1. Projeyi indir

GitHub'dan indiriyorsan:

```powershell
git clone REPO_URL
cd youtube-to-autoavsr-v0.7-
```

ZIP olarak indirdiysen klasoru ac ve PowerShell'de proje klasorune gir.

### 2. Sistem araclarini kur

PowerShell:

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg
winget install Git.Git
winget install GitHub.GitLFS
git lfs install
```

PowerShell'i kapatip yeniden ac.

### 3. Sanal ortami olustur ve paketleri kur

Proje klasorundeyken:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

### 4. Auto-AVSR, RetinaFace ve Whisper'i hazirla

```powershell
yt2avsr setup-external --config configs/default.yaml
yt2avsr setup-retinaface --config configs/default.yaml
yt2avsr setup-whisper --config configs/default.yaml
```

`setup-retinaface` sirasinda `ibug` paketleri Windows'ta kurulamazsa
`configs/default.yaml` icinde su satiri degistir:

```yaml
detector: mediapipe
```

Bu durumda RetinaFace yerine MediaPipe kullanilir.

### 5. Linkleri ekle ve calistir

Linkleri `sources_no_voiceover.txt` veya `sources_voiceover.txt` dosyasina yaz.

PowerShell'de sanal ortam aktifken:

```powershell
yt2avsr process-both-sources --config configs/default.yaml
```

Sanal ortam aktif degilse dogrudan:

```powershell
.venv\Scripts\yt2avsr.exe process-both-sources --config configs/default.yaml
```

## Tek Video veya Tek Liste Calistirma

Normal video:

```bash
ytavsr process "YOUTUBE_URL" --config configs/default.yaml
```

Playlist:

```bash
ytavsr process-playlist "PLAYLIST_URL" --config configs/default.yaml
```

Dis ses / dublaj olabilecek video:

```bash
ytavsr process "YOUTUBE_URL" --config configs/default.yaml --profile voiceover
```

Sadece tek bir kaynak dosyasini islemek:

```bash
ytavsr process-sources sources_voiceover.txt --config configs/default.yaml --profile voiceover
```

Windows'ta `ytavsr` yerine `yt2avsr` kullan:

```powershell
yt2avsr process "YOUTUBE_URL" --config configs/default.yaml
```

## Cikti Nerede Olusur?

Islenen klipler `data/clips/` altina yazilir:

```text
data/clips/<video>/<segment>/
├── source.mp4
├── mouth.mp4
├── audio.wav
├── transcript.txt
└── metadata.json
```

En onemli dosyalar:

| Dosya | Anlami |
| --- | --- |
| `mouth.mp4` | Auto-AVSR icin hazir 96x96 agiz videosu. |
| `transcript.txt` | Klip metni. |
| `metadata.json` | Klip hakkinda teknik bilgiler. |
| `source.mp4` | Ham klip, genelde kontrol/debug icin. |
| `audio.wav` | Klip sesi. |

Manifest dosyalari:

```text
data/manifests/
├── accepted.csv
├── review.csv
├── rejected.csv
└── all.csv
```

| Manifest | Anlami |
| --- | --- |
| `accepted.csv` | Egitim icin kullanilabilir klipler. |
| `review.csv` | Supheli, insan kontrolu iyi olur. |
| `rejected.csv` | Kullanilmamasi gereken klipler. |
| `all.csv` | Tum kayitlar. |

## Transcript Onceligi

Varsayilan akis:

```text
1. Creator tarafindan yuklenmis Turkce altyazi
2. Altyazi yoksa Whisper large-v3-turbo
3. Dusuk guvenli segmentleri eleme
```

YouTube otomatik altyazisi varsayilan olarak kapali. Cunku otomatik altyazi
tekrarli ve gurultulu olabiliyor. Bu ayar `configs/default.yaml` icindeki
`use_automatic_youtube_captions` ile degistirilebilir.

## Kalite Kontrolleri

Her klip su kontrollerden gecer:

- Agiz/yuz landmark gorunurlugu.
- Uzun sure agiz kaybi.
- Sahne gecisi / ani cut.
- Ses varken agzin hareket edip etmemesi.
- Landmark geometrisinin kararliligi.
- Whisper segment guveni.

Bu kontroller ten rengi, yuz rengi veya demografik ozellik kullanmaz. Geometrik
ve zamansal sinyallerle calisir.

## Profiller

Iki profil vardir:

| Profil | Kaynak dosya | Fark |
| --- | --- | --- |
| `no_voiceover` | `sources_no_voiceover.txt` | Sesin ekrandaki konusmaciya ait oldugu varsayilir. Lip-sync kontrolu daha toleranslidir. |
| `voiceover` | `sources_voiceover.txt` | Dis ses/dublaj olabilir. Ses varken agiz hareket etmiyorsa klip reddedilir. |

Dis ses iceren videolari mutlaka `sources_voiceover.txt` dosyasina koy.

## Sorun Giderme

### `ytavsr: command not found`

Terminal kisayolu henuz yuklenmemis olabilir:

```bash
source ~/.zshrc
```

Hala calismiyorsa proje klasorundeyken dogrudan calistir:

```bash
./.venv/bin/yt2avsr --help
```

### `yt2avsr: command not found`

Sanal ortam aktif degildir veya paket kurulumu yapilmamistir.

macOS / Linux:

```bash
./scripts/setup_once.sh
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install -e .
```

### `No module named 'ibug'`

RetinaFace bagimliliklari eksik demektir:

```bash
ytavsr setup-retinaface --config configs/default.yaml
```

Windows'ta kurulamiyorsa `configs/default.yaml` icinde:

```yaml
detector: mediapipe
```

### YouTube indirme hatasi

Indiriciyi test et:

```bash
ytavsr check-downloader "YOUTUBE_URL" --config configs/default.yaml
```

## Buluta Veri Yukleme: Hugging Face

Uretilen veri ortak private Hugging Face dataset reposuna yuklenebilir:

```text
https://huggingface.co/datasets/iboRotti/avsr-tr-dataset
```

Ilk kez giris yapmak icin:

1. Hugging Face -> Settings -> Access Tokens -> New token.
2. Token turu olarak `Write` sec.
3. Terminalde:

```bash
huggingface-cli login
```

Token'i terminale yapistir.

> Token'i repoya, koda veya herhangi bir dosyaya yazma.

Kendi verini yuklemek:

```bash
ytavsr push-data --config configs/default.yaml
```

Varsayilan olarak sadece `accepted` ve `review` klipleri yuklenir. Ham
`source.mp4` ve `audio.wav` yuklenmez.

Sesi de yuklemek istersen:

```bash
ytavsr push-data --config configs/default.yaml --include-audio
```

Ham kaynak videoyu da yuklemek istersen:

```bash
ytavsr push-data --config configs/default.yaml --include-source
```

Herkesin verisini indirmek:

```bash
ytavsr pull-data --config configs/default.yaml --dest data_cloud
```

Tek bir kisinin verisini indirmek:

```bash
ytavsr pull-data --config configs/default.yaml --dest data_cloud --contributor ibrahim
```

## Teknik Notlar

- Proje Python `>=3.10,<3.13` ister; pratikte Python 3.11 onerilir.
- Varsayilan Whisper modeli `large-v3-turbo`.
- Varsayilan dil `tr`.
- Varsayilan video yuksekligi en fazla 720p; guclu bilgisayarlar icin
  `configs/1080p.yaml` kullanilabilir.
- Auto-AVSR cikti boyutu 96x96.
- Varsayilan yuz tespit yontemi RetinaFace.
- Coklu yuz / aktif konusmaci secimi bu surumde kaldirildi.
- Sahne cut noktalari segmenti tamamen reddetmek yerine uygun yerde bolmek icin
  kullanilir.

## Lisans

Bu proje `Apache-2.0` lisansi ile dagitilir.
