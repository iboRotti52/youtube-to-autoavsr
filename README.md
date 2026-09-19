# YouTube -> Auto-AVSR Dataset Builder v0.7

Bu proje, izin verilen YouTube videolarindan Auto-AVSR tarzinda dudak okuma
egitim verisi uretir.

Kisaca yaptigi is:

1. YouTube videosunu indirir.
2. Whisper (`large-v3-turbo`) ile yuksek dogrulukta Turkce transcript uretir (YouTube altyazilari gurultulu oldugu icin varsayilan olarak kapali tutulur).
3. Videoyu cumle/segment parcalarina boler.
4. Kesilen her klibi ayrica Whisper ile tekrar dinleyerek transcript dogrulugu saglar.
5. Resmi Auto-AVSR preprocessing ile 96x96 agiz videosu (`mouth.mp4`) uretir.
6. Kotu veya supheli klipleri `accepted`, `review`, `rejected` olarak ayirir.

> Bu pipeline tek konusmacili, ekranda yuzu gorunen videolar icin tasarlanmistir.
> Panel, roportaj, coklu yuz veya ekranda konusmayan dis ses agirlikli videolar
> icin uygun degildir.

## Video Linkleri Nasıl Eklenir? (Önerilen Yöntem: `ytavsr add`)

Dosyaları elle açıp düzenlemek Git çakışmalarına (conflict) ve mükerrer linklere yol açabileceğinden, tüm link ekleme işlemleri için **`ytavsr add`** komutunu kullanın.

Bu komut otomatik olarak:
1. `git pull` yaparak ekip arkadaşlarının eklediği son linkleri çeker.
2. Eklenen linki Video ID bazında denetler (işlenmişse veya listede varsa uyarır, tekrar eklemez).
3. Linkteki gereksiz izleme ve oynatma listesi parametrelerini temizler.
4. Dosyaya ekleyip `git commit` ve `git push` ile GitHub'a otomatik gönderir (ekip anında güncel listeyi görür).

### Kullanım Örnekleri

```bash
# 1. Tek video eklemek (varsayılan: no_voiceover):
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID"

# 2. Birden fazla videoyu aynı anda eklemek:
ytavsr add "https://youtu.be/VID1" "https://youtu.be/VID2" "https://youtu.be/VID3"

# 3. Bir metin dosyasındaki tüm linkleri topluca eklemek:
ytavsr add yeni_linkler.txt

# 4. Dış ses / dublaj / anlatıcı içeren videolar için:
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID" --voiceover

# 5. Sadece yerel dosyaya eklemek (GitHub'a hemen push yapmamak):
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID" --no-push
```

### Video Profilleri ve Kaynak Dosyaları

Sistem videoları türüne göre iki dosyaya ayırır:

| Profil | Dosya | Ekleme Komutu | Ne zaman kullanılır? |
| --- | --- | --- | --- |
| `no_voiceover` | `sources_no_voiceover.txt` | `ytavsr add "URL"` | Ekrandaki kişi konuşuyor, dış ses/dublaj yok (varsayılan). Lip-sync esnektir. |
| `voiceover` | `sources_voiceover.txt` | `ytavsr add "URL" --voiceover` | Anlatıcı, dış ses, dublaj olabilir. Ses varken ağız oynamıyorsa klip elenir. |
| Takip | `processed_sources.txt` | Otomatik güncellenir | Daha önce işlenmiş videolar. Otomatik olarak atlanır. |

> **Manuel Düzenleme:** Dosyaları doğrudan metin editörüyle düzenlediyseniz veya toplu yapıştırma yaptıysanız, mükerrerleri temizlemek için `ytavsr dedup-sources` komutunu çalıştırabilirsiniz.

## En Kisa Kullanim

Kurulumdan sonra sadece sunu calistir:

```bash
ytavsr
```

Bu komut:

- `sources_no_voiceover.txt` dosyasindaki linkleri `no_voiceover` profiliyle,
- `sources_voiceover.txt` dosyasindaki linkleri `voiceover` profiliyle

isler. Bos dosyalari atlar. Otomatik olarak liste ici ve listeler arasi mukerrer linkleri filtreler.

### Mukerrer (Duplicate) Video Korumasi

Sistem video linklerini video kimligi (Video ID) bazinda otomatik olarak tekillestirir:
- **Otomatik Calisma Korumasi:** `ytavsr` komutu calistiginda; ayni dosya icindeki veya iki dosya arasindaki mukerrer linkler (farkli URL formatlari `youtu.be`, `shorts`, veya `&list=...&index=...` gibi parametreler icerse dahi) otomatik olarak tespit edilir ve sadece bir kez islenir.
- **Kaynak Dosyalarini Temizlemek:** Kaynak dosyalarindaki tekrarlayan satirlari temizlemek icin:
  ```bash
  # Sadece kontrol etmek ve mukerrerleri listelemek:
  ytavsr dedup-sources --check

  # Dosyalardaki mukerrerleri yerinde temizlemek (yorum ve siralamayi korur):
  ytavsr dedup-sources
  ```

## Coklu Bilgisayar ile Paralel Calisma (Sharding / Is Bolusturme)

Ayni anda birden fazla kisi (veya bilgisayar) ayni kaynak dosyasi uzerinde calisacaksa, ayni videolarin mukerrer islenmemesi icin `--shard <index>/<total>` parametresini kullan:

Ekip üyelerinin shard dağılımı sabittir (asla değişmez):
- **İbrahim Gözlükaya (Shard 0):** `ytavsr --shard 0/3` *(veya `--shard ibrahim-gozlukaya` ya da `--shard 0`)*
- **Damla Kemal (Shard 1):** `ytavsr --shard 1/3` *(veya `--shard damla` ya da `--shard 1`)*
- **İbrahim Billurcu (Shard 2):** `ytavsr --shard 2/3` *(veya `--shard ibrahim-billurcu` ya da `--shard 2`)*


Bu ayar sayesinde:
- Kaynak listesindeki tekil videolar ve buyuk YouTube oynatma listeleri (playlist) 3 bilgisayar arasinda esit bolunur.
- Hicbir bilgisayar ayni videoyu indirip islemez (mukerrer islem ve zaman kaybi yasanmaz).
- Her calistirmada Hugging Face'teki son durum otomatik kontrol edilip onceden islenmis olanlar atlanir.

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

## Modal ile Bulutta GPU İşleme (RetinaFace + 1080p)

Bilgisayarınızı meşgul etmeden ve Mac CPU'suyla uğraşmadan **RetinaFace + 1080p** resmi Auto-AVSR akışını sunucusuz NVIDIA T4 GPU üzerinde çalıştırmak için **Modal** entegrasyonu kullanılır.

> Modal her kullanıcıya **aylık 30$ ücretsiz GPU kredisi** sağlar. Ekipteki her arkadaşınız kendi kişisel Modal hesabı üzerinden çalışabilir.

### 1. Ön Koşul: Hugging Face Girişi (Bir Kez)
İşlenen kliplerin doğrudan Hugging Face deposuna aktarılabilmesi için bilgisayarınızda oturum açılmış olmalıdır:
```bash
huggingface-cli login
```
*(Token'ınız yoksa: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) adresinden 'Write' yetkili bir token alın).*

### 2. Modal Kurulumu (Bir Kez)
```bash
pip install modal
modal setup
```

### 3. Çalıştırma
```bash
# Listedeki tüm bekleyen videoları bulut GPU'sunda işle ve HF'ye gönder:
ytavsr modal

# Ekip üyeleriyle sabit shard paylaşımı (Asla değişmez):
ytavsr modal --shard 0/3   # İbrahim Gözlükaya (veya --shard ibrahim-gozlukaya / --shard 0)
ytavsr modal --shard 1/3   # Damla Kemal       (veya --shard damla / --shard 1)
ytavsr modal --shard 2/3   # İbrahim Billurcu  (veya --shard ibrahim-billurcu / --shard 2)


# Tek video test etmek:
ytavsr modal --url "https://www.youtube.com/watch?v=VIDEO_ID"
```
Üretilen 96×96 ağız ROI videoları (`mouth.mp4`), Whisper `large-v3-turbo` metinleri ve kalite manifestoları doğrudan Hugging Face deposuna yüklenir; ev/ofis internetinizin yükleme kotası ve yerel diskiniz harcanmaz.


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

Videolari projeye eklemek icin (otomatik git pull, mukerrer kontrolu ve push ile):

```bash
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID"
```

Ardindan veri uretimini baslat:

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

Videolari projeye eklemek icin:

```powershell
yt2avsr add "https://www.youtube.com/watch?v=VIDEO_ID"
```

PowerShell'de sanal ortam aktifken veri uretimini baslat:

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
├── clip_transcript.json
├── transcript.txt
└── metadata.json
```

En onemli dosyalar:

| Dosya | Anlami |
| --- | --- |
| `mouth.mp4` | Auto-AVSR icin hazir 96x96 agiz videosu. |
| `transcript.txt` | Klip metni. |
| `clip_transcript.json` | Klip sesinden yeniden uretilen kelime zamanlari ve guven skorlari. |
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
1. Ana videoyu Whisper large-v3-turbo ile yazıya dök
2. Bu metinden klip sınırlarını seç
3. Kesilmiş her klibi ayrıca Whisper ile yazıya dök
4. transcript.txt dosyasını klip sesinden çıkan metinle güncelle
5. Eski/yeni metin benzerliği düşükse klibi review grubuna gönder
6. Düşük güvenli segmentleri ele
```

YouTube altyazilari (hem manuel hem otomatik) varsayilan olarak kapalidir (`use_youtube_subtitles: false`). Cunku YouTube altyazilari tekrarli, kayan pencereli veya senkron kaymali olabiliyor ve egitim etiketleri icin gurultu olusturabiliyor. Bunun yerine tum akista dogrudan Whisper (`large-v3-turbo`) kullanilir. Istenirse bu ayarlar `configs/default.yaml` icindeki `use_youtube_subtitles` ve `use_automatic_youtube_captions` ile degistirilebilir.

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

Dis ses iceren videolari eklerken her zaman `--voiceover` parametresini kullan: `ytavsr add "URL" --voiceover`.

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

Model eğitimi ve Türkçe Auto-AVSR fine-tune akışı ayrı
`auto-avsr-turkish-finetune` projesinde tutulur.

Tek bir kisinin verisini indirmek:

```bash
ytavsr pull-data --config configs/default.yaml --dest data_cloud --contributor ibrahim
```

### Daha Once Islenmis Videolari Senkronize Etmek (Mukerrer Onleme)

Ekipteki diger arkadaslarinizin Hugging Face'e yukledigi tum videolari tarayip kendi `processed_sources.txt` dosyaniza eklemek icin:

```bash
ytavsr sync-processed --config configs/default.yaml
```

Bu komut sayesinde baska birinin isledigi videolar otomatik olarak listenize eklenir ve sizde tekrar indirilip islenmez.

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
