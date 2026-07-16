# YouTube → Auto-AVSR Dataset Builder v0.7

Bu araç, izin verilen YouTube videolarından Auto-AVSR tarzı dudak-okuma eğitim
verisi üretir. Öne çıkan noktalar:

1. **Resmî Auto-AVSR preprocessing** kullanılır: yüz tespiti → landmark → mean-face
   hizalama → 96×96 ağız ROI.
2. **Tek konuşmacı (talking-head) varsayılır.** Çoklu yüz takibi / aktif konuşmacı
   seçimi yoktur; ekrandaki yüzü resmî Auto-AVSR cropper seçer. Dış ses/dublaj,
   `voiceover` profilinde dudak-ses (lip-sync) kontrolüyle elenir.
3. Creator tarafından yüklenmiş Türkçe altyazı varsa o kullanılır; yoksa
   `faster-whisper large-v3-turbo` çalışır. Düşük güvenli Whisper segmentleri reddedilir.

## Kurulum (macOS / Linux)

```bash
brew install ffmpeg git git-lfs      # Linux: apt install ffmpeg git git-lfs
git lfs install
./scripts/setup_once.sh
./scripts/install_terminal_command.sh
source ~/.zshrc
```

Bundan sonra bu klasore girmeden terminalden `ytavsr` yazman yeterli olur.
`ytavsr-setup` ise kurulumu tekrar hazirlamak/guncellemek gerektiginde kullanilir.

Eger Terminal `yt2avsr: command not found` derse sorun veri isleme degil, komutun
global olarak kurulu olmamasidir. Bu projede disaridan kullanilacak komut
`ytavsr`'dir. Alternatif olarak proje klasorundeyken sanal ortamdaki komutu
dogrudan calistirabilirsin:

```bash
./.venv/bin/yt2avsr --help
```

Windows'ta ayni dogrudan komut yolu farklidir:

```powershell
.venv\Scripts\yt2avsr.exe --help
```

## Kurulum (Windows)

Aynı adımlar, sadece sistem araçları ve venv aktivasyonu farklı. **Python 3.11
kullan** (mediapipe 0.10.x Windows'ta 3.12+ ile sorun çıkarabilir).

```powershell
# 1) Sistem araçları (PowerShell)
winget install Gyan.FFmpeg
winget install GitHub.GitLFS
git lfs install

# 2) Sanal ortam + paket
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3) Auto-AVSR + RetinaFace
yt2avsr setup-external --config configs/default.yaml
yt2avsr setup-retinaface --config configs/default.yaml
yt2avsr setup-whisper --config configs/default.yaml
```

> Windows'ta `setup-retinaface` sırasında `ibug` paketleri kurulamazsa, o makinede
> `configs/default.yaml` içinde `detector: mediapipe` yap — ibug/torch gerektirmez,
> aynı 96×96 ağız ROI'sini üretir. Ana veriyi RetinaFace ile üretmeye devam
> edebilirsin; yalnızca o makine mediapipe ile çalışır.

`setup-external`, resmî `mpc001/auto_avsr` reposunu `external/auto_avsr` altına klonlar
ve preprocessing bağımlılıklarını kurar.

`setup-retinaface`, resmî Auto-AVSR'ın kullandığı RetinaFace yüz tespit/hizalama
paketlerini kurar: `torch` + `ibug.face_detection` + `ibug.face_alignment`
(önceden eğitilmiş ağırlıklar Git LFS ile iner, bu yüzden `git-lfs` şarttır).
Bu adım olmadan `detector: retinaface` ile crop çalışmaz (`No module named 'ibug'`).

`setup-whisper`, config'teki Whisper modelini (varsayılan `large-v3-turbo`) önceden
indirir. Böylece altyazısı olmayan ilk videoda model indirmesi için beklemezsin.
Model bir kez indirilip Hugging Face önbelleğine kaydedilir.

## Çalıştırma

Ana kullanım: linkleri kaynak (`sources`) dosyalarına yaz, tek komutla işle.

1. Linkleri (satır başına bir tane; `#` ile başlayan satırlar yorum) doğru dosyaya ekle:
   - `sources_no_voiceover.txt` → dış sesi olmayan, ekranda konuşan videolar
   - `sources_voiceover.txt` → dış ses / anlatıcı / dublaj içerebilen videolar
2. Çalıştır:

```bash
ytavsr
```

İki dosyadan hangisinde link varsa o işlenir; boş dosya atlanır. Tek bir listeyi
işlemek istersen:

```bash
ytavsr process-sources sources_voiceover.txt --config configs/default.yaml --profile voiceover
```

Hızlı test için tek video veya playlist:

```bash
ytavsr process "YOUTUBE_URL" --config configs/default.yaml
ytavsr process-playlist "PLAYLIST_URL" --config configs/default.yaml
ytavsr process "YOUTUBE_URL" --config configs/default.yaml --profile voiceover
```

## Transcript önceliği

```text
creator tarafından yüklenmiş tr/tr-TR altyazı
    ↓ yoksa
Whisper large-v3-turbo + word timestamps + VAD
    ↓
segment confidence filtresi
```

YouTube'un otomatik caption'ı varsayılan olarak kullanılmaz. Çünkü o da otomatik bir
ASR etiketidir ve model/sürüm/kalite kontrolü bizim elimizde değildir. İstenirse config
üzerinden daha sonra ayrı bir fallback olarak eklenebilir.

## Aktif konuşmacı (kaldırıldı)

Eski sürümlerdeki `av_sync` çoklu yüz takibi ve TalkNet tabanlı "hangi yüz
konuşuyor" seçimi **kaldırılmıştır**. Pipeline artık tek konuşmacılı
(talking-head) video varsayar; yüz seçimini doğrudan resmî Auto-AVSR cropper yapar.
Dış ses/dublaj ise `voiceover` profilinde dudak-ses (lip-sync) kontrolüyle elenir
(bkz. "Profiller: iki source, tek fark"). Röportaj/panel gibi çok konuşmacılı
videolar bu repo için uygun değildir.

## Whisper doğruluğu

Whisper için tek bir evrensel “doğruluk oranı” garanti edilemez; Türkçe WER, mikrofon,
aksan, arka plan müziği ve konuya göre değişir. Bu repo doğruluğu şu şekilde korur:

- `large-v3-turbo` (gürültülü/ağır aksanlı içerik için `large-v3`)
- dil zorlaması: `tr`
- VAD
- temperature 0
- word probability
- segment log-probability/no-speech kontrolü
- `min_asr_confidence: 0.72`
- düşük güvenli klipleri `accepted.csv` dışında bırakma

Gerçek yeterlilik ölçümü için 30–60 dakikalık temsili bir örneği elle transkribe edip WER
ölçmek gerekir. Model eğitimi için hedef olarak başlangıçta **WER ≤ %10–15** ve ardından
manuel örnekleme önerilir.

## Çıktı

```text
data/clips/<video>/<segment>/
├── source.mp4
├── mouth.mp4
├── audio.wav
├── transcript.txt
└── metadata.json
```

`mouth.mp4`, resmî Auto-AVSR hizalama/crop akışından çıkar.


## `sources.txt` ile toplu çalışma

Repo kökündeki `sources.txt` dosyasını aç:

```text
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://www.youtube.com/playlist?list=PLAYLIST_ID
```

Açıklama yazmak için `#` kullanabilirsin:

```text
# Tek konuşmacılı eğitim videoları
https://www.youtube.com/watch?v=VIDEO_ID_1

# Bu bir playlist
playlist https://www.youtube.com/playlist?list=PLAYLIST_ID
```

Sonra yalnızca şu komutu çalıştır:

```bash
yt2avsr process-sources sources.txt --config configs/default.yaml
```

Bir link hata verirse diğer linklerle devam eder. Başarılı aşamalar SQLite içinde
saklandığı için komutu tekrar çalıştırdığında tamamlanan işler yeniden yapılmaz.


## v0.4: ağız kapanması, kupa/el ve araya görsel girmesi

Her klip artık üç sınıftan birine gider:

```text
accepted.csv  → net biçimde kullanılabilir
review.csv    → sınırda; otomatik silinmez, insan kontrol eder
rejected.csv  → dudak/yüz uzun süre yok, ciddi sahne değişimi veya konuşma-dudak uyumsuzluğu
```

Kontroller:

- ağız landmark'larının segment boyunca bulunma oranı,
- en uzun kesintisiz ağız kaybı,
- ani sahne değişimleri,
- ses aktifken dudak hareketinin uzun süre çok düşük kalması,
- landmark geometrisindeki ani sıçramalar.

Bu kurallar **ten rengi, yüz rengi veya demografik özellik kullanmaz**. Yalnızca geometrik
ve zamansal sinyaller kullanılır. Yanlış eleme ve veri bias'ını azaltmak için eşikler iki
kademelidir: kesin kötü klipler reddedilir, belirsiz klipler `review.csv` içine alınır.

Varsayılan ayarlar bilinçli olarak muhafazakârdır:

```yaml
visual_quality:
  accept_min_mouth_visible_ratio: 0.88
  review_min_mouth_visible_ratio: 0.68
  accept_max_missing_run_seconds: 0.35
  review_max_missing_run_seconds: 1.00
```

Dolayısıyla birkaç karelik landmark kaybı klibi doğrudan silmez. Kupa ağzı uzun süre
kapattığında, slayt/stock görüntü girdiğinde veya ses varken dudak hareketi kaybolduğunda
klip `review` ya da `rejected` olur.


## v0.5: YouTube EJS challenge otomatik çözümü

Artık elle `python -m yt_dlp --remote-components ...` çalıştırmak gerekmez. Pipeline,
yt-dlp Python API'sine otomatik olarak şunları verir:

```text
remote_components = {"ejs:github"}
js_runtimes = {"deno": {}}
```

Ayrıca format seçimi otomatik olarak üç kez denenir:

```text
config'teki format
→ bestvideo[height<=1080]+bestaudio/best[height<=1080]
→ bv*+ba/b
```

Test:

```bash
yt2avsr check-downloader \
  "https://www.youtube.com/watch?v=LF0L-T-EDQI" \
  --config configs/default.yaml
```

Toplu çalışma yine tek komuttur:

```bash
yt2avsr process-sources sources.txt --config configs/default.yaml
```

Bir kaynak başarısız olursa diğer kaynaklar işlenir; ancak komut artık yanlış biçimde
`Done` yazmaz. En sonda başarısız linkleri gösterip hata koduyla kapanır.


## v0.6: hızlı altyazı ve görsel preprocessing

Transcript sırası artık:

```text
1. creator tarafından yüklenmiş Türkçe altyazı
2. YouTube otomatik Türkçe altyazısı
3. ikisi de yoksa faster-whisper large-v3-turbo
```

Bu nedenle altyazısı bulunan videolarda Whisper hiç yüklenmez ve transkripsiyon süresi
neredeyse tamamen ortadan kalkar. Manifestte kaynak ayrı tutulur:

```text
youtube_manual_subtitles
youtube_auto_captions
whisper
```

Görsel işlem hızlandırmaları:

- Resmî Auto-AVSR landmark detector ve crop processor her segment için tekrar
  oluşturulmaz; bir kez belleğe yüklenir ve klipler arasında yeniden kullanılır.
- Kesin reddedilecek görsel kliplerde resmî Auto-AVSR crop çalıştırılmaz.

> Not: v0.7'de çoklu yüz aktif konuşmacı seçimi (ASD / av_sync / TalkNet) tamamen
> kaldırıldı. Bu nedenle eski sürümlerdeki "tek-yüz bypass" ve "her 5 karede bir ASD"
> davranışları artık geçerli değildir; pipeline tek konuşmacı varsayar.

Çalıştırma:

```bash
yt2avsr process-sources sources.txt --config configs/default.yaml
```


## Profiller: iki source, tek fark

Sadece iki kaynak listesi vardır ve aralarındaki **tek fark videoda dış ses
(voice-over) olup olmamasıdır**. Her iki listede de AYNI işlemler uygulanır:

- YouTube manuel/otomatik altyazı fallback + Whisper large-v3-turbo fallback
- segmentasyon
- ağız landmark görünürlüğü (dudak görünümü kontrolü)
- scene-cut kontrolü
- uzun süreli ağız kaybı
- occlusion/landmark kararsızlığı
- resmî Auto-AVSR crop
- accepted/review/rejected ayrımı

Çoklu yüz / "ekranda hangi kişi konuşuyor" mantığı (TalkNet ve av_sync aktif
konuşmacı) **kaldırılmıştır**. Tek konuşmacılı (talking-head) video varsayılır;
yüz seçimini resmî Auto-AVSR cropper yapar.

Tek fark:

```text
sources_no_voiceover.txt  (dış sessiz)
→ ses, ekrandaki konuşmacıya ait kabul edilir
→ dudak-ses (lip-sync) kontrolü gevşetilir; doğal duraklamalar segmenti düşürmez

sources_voiceover.txt     (dış sesli)
→ dış ses / dublaj olabilir
→ dudak-ses kontrolü sıkı uygulanır: ses varken ağız kıpırdamıyorsa
  segment "external_voice_or_dubbing" nedeniyle reddedilir
```

Bu kontrol `visual_quality`'nin `static_speech_ratio` sinyalidir (ses aktifken
ağzın hareket etmediği kare oranı) — ayrı bir model gerekmez.

Çalıştırma:

```bash
yt2avsr process-both-sources --config configs/default.yaml
```

**Önemli:** Dış ses (anlatıcı / dublaj) içeren videolar mutlaka
`sources_voiceover.txt`'de olmalı. Dış ses cümleleri transkriptte görünse bile
o kliplerin sesi ekrandaki ağızla eşleşmediği için lip-sync kontrolü onları
`external_voice_or_dubbing` ile reddeder; veri setine ulaşmaz.


## v0.7: tekrar, dış ses ve cut düzeltmeleri

Üç veri kalitesi sorunu giderildi:

**1) Altyazı tekrarları.** YouTube oto-altyazısı "kayan pencere" formatında gelir
(her cue öncekini tekrarlayıp kelime ekler) ve `>>` konuşmacı işaretleri içerir.
Varsayılan artık `use_automatic_youtube_captions: false` — insan yapımı manuel
altyazı yoksa temiz kelime-zamanlı Whisper etiketi kullanılır. Ayrıca
`subtitles.py` içindeki `deduplicate()` kelime-seviyesi *rolling* örtüşme
temizliğine yükseltildi ve `>>` işaretleri metinden atılır (manuel altyazı hâlâ
kullanıldığında da güvenli).

**2) Dış ses cümleleri.** Anlatıcıyı sesten ayırt etmek transkript metninden
mümkün değildir; doğru kaldıraç `voiceover` profilinin lip-sync kontrolüdür.
Narrated videolar `sources_voiceover.txt`'ye konur; ses aktifken ağız
oynamıyorsa klip reddedilir. `>>` temizliği de metni sadeleştirir.

**3) Cut'lar akışı bozuyor.** Yeni `scenes.py` (ffmpeg `select='gt(scene,thr)'`)
sahne-kesme zamanlarını çıkarır; `make_segments` bir segmenti reddetmek yerine
**kesme noktasında böler**, kesmenin iki yanındaki temiz (sürekli dudak
hareketli) parçaları korur (`min_duration`'ı sağlayanlar). Ayar:
`segmentation.split_on_scene_cut` (varsayılan `true`) ve `scene_cut_threshold`
(0.0–1.0; düşük değer daha agresif böler).


## Bulut veri paylaşımı (Hugging Face)

Kod GitHub'da, **üretilen eğitim verisi** ise ortak bir *private* Hugging Face
dataset repo'sunda durur. Her ekip üyesi kendi videolarını lokalde işler, sonra
verisini buluta gönderir. Herkesin verisi `data/<kullanıcı>/...` altına yazıldığı
için kimse birbirinin klibinin üstüne yazmaz.

### Veri seti

Ortak (private) veri seti reposu:

```text
https://huggingface.co/datasets/iboRotti/avsr-tr-dataset
```

Bu değer `configs/default.yaml` içinde ayarlıdır ve herkes AYNI değeri kullanır:

```yaml
cloud:
  repo_id: "iboRotti/avsr-tr-dataset"
  private: true
```

Repoya erişim: sahibi (iboRotti) her ekip üyesini HF'de
dataset → **Settings → Collaborators**'tan **write** yetkisiyle davet eder.

### Token (kimlik doğrulama)

Yüklemek/indirmek için her üye kendi HF **token**'ı ile bir kez giriş yapar:

1. HF → **Settings → Access Tokens → New token** → tür **Write** → oluştur.
2. Terminal'de:

   ```bash
   huggingface-cli login        # token'ı buraya yapıştır
   ```

> ⚠️ **Token'ı asla bu repoya, koda veya herhangi bir dosyaya yazma.** Token
> şifre gibidir; sadece `huggingface-cli login` ile kendi makinende saklanır.
> `.gitignore` `.env` / `*.token` dosyalarını zaten dışlar. 

Kendi verini yükle (varsayılan olarak yalnızca `accepted` + `review` klipleri):

```bash
ytavsr push-data --config configs/default.yaml
```

Windows'ta sanal ortam aktifse ayni islem:

```powershell
yt2avsr push-data --config configs/default.yaml
```

Eger `ytavsr` bulunamazsa once Terminal ayarini tekrar yukle:

```bash
source ~/.zshrc
```

Ya da proje klasorundeyken komutu dogrudan sanal ortamdan calistir:

```bash
./.venv/bin/yt2avsr push-data --config configs/default.yaml
```

Windows'ta dogrudan sanal ortamdan calistirma:

```powershell
.venv\Scripts\yt2avsr.exe push-data --config configs/default.yaml
```

Buluta yalnızca görsel dudak-okuma (VSR) için gerekenler gider: `mouth.mp4`,
`transcript.txt`, `metadata.json`. Ham `source.mp4` (büyük, sadece hata ayıklama)
ve `audio.wav` **yüklenmez**. Sesli-görüntülü (AV) model için sesi de istersen
`--include-audio`, ham klibi de istersen `--include-source` ekle. HF ücretsiz
hesapta private depolama 100 GB'dır.

Eğitim makinesinde herkesin verisini tek seferde indir:

```bash
yt2avsr pull-data --config configs/default.yaml --dest data_cloud
```

İnen yapı doğrudan eğitime hazırdır:

```text
data_cloud/
└── data/
    ├── ibrahim/
    │   ├── clips/<video>/<segment>/{mouth.mp4,transcript.txt,metadata.json}
    │   └── manifests/{all.csv,accepted.csv,...}
    ├── arkadas2/
    └── arkadas3/
```

Sadece kabul edilenleri yüklemek istersen: `--include accepted`.
Tek bir kişinin verisini çekmek: `yt2avsr pull-data --contributor ibrahim`.
