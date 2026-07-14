# YouTube → Auto-AVSR Dataset Builder v0.2

Bu sürüm üç önemli değişiklik getirir:

1. **Resmî Auto-AVSR preprocessing** kullanılır: yüz tespiti → landmark → mean-face
   hizalama → 96×96 ağız ROI.
2. Birden fazla yüz varsa yüzler takip edilir ve sesle eşzamanlı ağız hareketi en yüksek
   olan track aktif konuşmacı olarak seçilir.
3. Creator tarafından yüklenmiş Türkçe altyazı varsa o kullanılır; yoksa
   `faster-whisper large-v3-turbo` çalışır. Düşük güvenli Whisper segmentleri reddedilir.

## Kurulum

```bash
brew install ffmpeg git
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
yt2avsr setup-external --config configs/default.yaml
```

`setup-external`, resmî `mpc001/auto_avsr` reposunu `external/auto_avsr` altına klonlar
ve preprocessing bağımlılıklarını kurar.

## Çalıştırma

```bash
yt2avsr process "YOUTUBE_URL" --config configs/default.yaml
```

Playlist:

```bash
yt2avsr process-playlist "PLAYLIST_URL" --config configs/default.yaml
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

- 40 seyrek karede yalnızca tek yüz görülürse aktif konuşmacı seçimi tamamen atlanır.
- Çok yüzlü videolarda ASD her kare yerine varsayılan olarak her 5 karede bir, 480 px
  analiz çözünürlüğünde çalışır.
- Resmî Auto-AVSR landmark detector ve crop processor her segment için tekrar
  oluşturulmaz; bir kez belleğe yüklenir ve klipler arasında yeniden kullanılır.
- Kesin reddedilecek görsel kliplerde resmî Auto-AVSR crop çalıştırılmaz.

Talking-head videolarında en büyük kazanç tek-yüz bypass'ından gelir. Röportaj/panel
videolarında ASD devam eder fakat seyrek ve düşük çözünürlüklü analiz nedeniyle daha hızlıdır.

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


## Bulut veri paylaşımı (Hugging Face)

Kod GitHub'da, **üretilen eğitim verisi** ise ortak bir *private* Hugging Face
dataset repo'sunda durur. Her ekip üyesi kendi videolarını lokalde işler, sonra
verisini buluta gönderir. Herkesin verisi `data/<kullanıcı>/...` altına yazıldığı
için kimse birbirinin klibinin üstüne yazmaz.

Bir kereye mahsus giriş:

```bash
huggingface-cli login        # HF token'ını yapıştır (Settings → Access Tokens)
```

`configs/default.yaml` içindeki `cloud.repo_id` alanını ortak repoya ayarla
(herkes AYNI değeri kullanır):

```yaml
cloud:
  repo_id: "takim-adi/avsr-tr-dataset"
  private: true
```

Kendi verini yükle (varsayılan olarak yalnızca `accepted` + `review` klipleri):

```bash
yt2avsr push-data --config configs/default.yaml
```

Eğitim makinesinde herkesin verisini tek seferde indir:

```bash
yt2avsr pull-data --config configs/default.yaml --dest data_cloud
```

İnen yapı doğrudan eğitime hazırdır:

```text
data_cloud/
└── data/
    ├── ibrahim/
    │   ├── clips/<video>/<segment>/{mouth.mp4,audio.wav,transcript.txt,metadata.json}
    │   └── manifests/{all.csv,accepted.csv,...}
    ├── arkadas2/
    └── arkadas3/
```

Sadece kabul edilenleri yüklemek istersen: `--include accepted`.
Tek bir kişinin verisini çekmek: `yt2avsr pull-data --contributor ibrahim`.
