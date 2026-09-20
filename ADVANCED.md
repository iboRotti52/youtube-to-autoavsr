# 🛠️ Gelişmiş Komutlar ve Detaylı Seçenekler (ADVANCED)

Bu belge, günlük standart iş akışı dışındaki gelişmiş parametreleri, alternatif çalıştırma yöntemlerini ve teknik detayları içerir. Günlük ekip kullanımı için ana rehber olan [README.md](README.md) dosyasına bakabilirsiniz.

---

## 📌 Video Profilleri ve Kaynak Dosyaları

Projede iki tür video profili desteklenmektedir:

1. **`no_voiceover` (`sources_no_voiceover.txt`) [ŞU AN AKTİF OLAN]:**
   - Konuşmacının yüzü ekranda doğrudan görünür, kamera açısı konuşmacıya dönüktür.
   - Dış ses, çevirmen dublajı veya arkadan anlatıcı içermez.
   - Ağız hareketi ile ses arasındaki doğal duraklamalara karşı daha toleranslıdır.
   - **Şu an ekibimizin topladığı ve eğiteceği ana veri seti budur.**

2. **`voiceover` (`sources_voiceover.txt`) [İLERİ AŞAMA]:**
   - Belgesel, dış sesli haberler veya dublajlı videolar içindir.
   - Katı dudak-ses senkronizasyonu (lip-sync) filtresi uygular; ekranda ağız oynamıyorken ses geliyorsa o parçayı eler (`rejected`).

---

## ➕ Gelişmiş Link Ekleme Seçenekleri (`ytavsr add`)

Normalde `ytavsr add "URL"` veya `ytavsr add dosya.txt` komutu doğrudan `sources_no_voiceover.txt` dosyasına yazar ve otomatik olarak `git pull` & `git push` yapar. İhtiyaç halinde aşağıdaki bayraklar kullanılabilir:

| Bayrak | Açıklama |
| :--- | :--- |
| `--voiceover`, `-vo` | Linki `sources_voiceover.txt` dosyasına ekler. |
| `--target`, `-t <dosya>` | Özel bir hedef metin dosyası belirtir (örn: `-t ozel_liste.txt`). |
| `--file`, `-f <dosya>` | Linklerin okunacağı dosyayı açıkça belirtir (örn: `-f linkler.txt`). |
| `--no-push` | Linki ekler ancak GitHub'a otomatik `git push` yapmaz. |
| `--keep-raw` | YouTube linklerindeki playlist, indeks vb. parametreleri temizlemeden olduğu gibi ekler. |

### Örnekler:
```bash
# Dış sesli bir videoyu voiceover listesine ekleme:
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID" --voiceover

# Dosyadan topluca voiceover listesine ekleme:
ytavsr add belgeseller.txt --voiceover

# Ekleyip henüz GitHub'a göndermek istemiyorsanız:
ytavsr add "https://youtu.be/VID" --no-push
```

---

## ☁️ Gelişmiş Modal GPU Seçenekleri (`ytavsr modal`)

Modal üzerinde çalışırken kullanılabilecek ek parametreler:

```bash
# Listeye eklemeden tek bir videoyu Modal GPU'da doğrudan işlemek:
ytavsr modal --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Tek bir videoyu voiceover profili ile test etmek:
ytavsr modal --url "https://www.youtube.com/watch?v=VIDEO_ID" --voiceover

# Özel shard oranı belirtmek (örn: 5 parçaya bölüp 2. parçayı işlemek):
ytavsr modal --shard 2/5

# Tamamlanmış aşamaları zorla yeniden çalıştırmak:
ytavsr modal --shard 0 --force
```

---

## 💻 Yerel (Local) Çalıştırma Seçenekleri

Kendi bilgisayarınızda (GPU veya Mac CPU) çalıştırırken kullanabileceğiniz komutlar.
`--config` vermezseniz lokal komutlar `configs/default.yaml` davranışını kullanır (detector: **MediaPipe**):

```bash
# Belirli bir yerel video dosyasını veya klasörünü işlemek:
ytavsr process-local /yol/video.mp4
ytavsr process-local /yol/videolar_klasoru/
# (Klasör verilirse içindeki .mp4/.mkv/.webm/.mov/.m4v dosyalarının hepsi
# alfabetik sırayla işlenir; başka dosyalar atlanır. Boş/uygunsuz klasörde
# anlaşılır bir hata verilir.)

# Farklı bir kaynak dosyasını yerelde işlemek:
ytavsr process-sources sources_voiceover.txt --profile voiceover

# RetinaFace ile çalıştırmak (güçlü GPU; torch + ibug gerekir):
ytavsr setup-retinaface --config configs/retina_1080p.yaml  # bir kez
ytavsr process-local /yol/video.mp4 --config configs/retina_1080p.yaml

# Özel yapılandırma dosyası (YAML) ile çalıştırmak:
ytavsr --config configs/custom.yaml --shard 0

# Tamamlanan aşamaları zorla baştan çalıştırmak:
ytavsr --force
```

**Lokal varsayılanlar (Mac):**
- Detector: `mediapipe` (RetinaFace kurulumu gerekmez).
- Whisper (`faster-whisper` MPS desteklemez): `auto` → CUDA varsa `cuda/float16`, yoksa `cpu/int8`. Açıkça `mps` yazılırsa uyarıyla `cpu/int8`'e düşülür.
- İlk çalıştırmada kullanılan `detector`, `auto_avsr_device`, `whisper_device/compute/model` logda tek satır olarak görünür.
- Eksik bağımlılıklar anlaşılır hata verir: `ffmpeg` yoksa kurulum komutuyla, Auto-AVSR yoksa `setup-external` komutuyla, Whisper modeli inemezse `setup-whisper` komutuyla yönlendirilir.

---

## 🔄 Veri & Liste Senkronizasyon Komutları

| Komut | Açıklama |
| :--- | :--- |
| `ytavsr sync-processed` | Hugging Face'teki (`avsr-tr-ekip/avsr-tr-dataset`) en güncel `processed_sources.txt` dosyasını çekip yerel liste ile birleştirir. |
| `ytavsr push-data` | Yerelde işlenmiş ve kabul edilmiş (`accepted`) klipleri Hugging Face deposuna yükler. |
| `ytavsr pull-data` | Model eğitimi yapacak sunucuda Hugging Face'teki tüm ekip verisini yerel diske indirir. |
| `ytavsr dedup-sources` | Kaynak dosyalarındaki mükerrer satırları temizler. |
| `ytavsr dedup-sources --check` | Dosyayı değiştirmeden sadece mükerrer satır olup olmadığını raporlar. |
| `ytavsr inspect-data` | Üretilen kliplerin, sürelerin ve kabul/ret oranlarının özet istatistiklerini terminalde gösterir. |

---

## ⚙️ Yapılandırma (`configs/default.yaml`)

Tüm pipeline eşik değerleri [configs/default.yaml](configs/default.yaml) dosyasından yönetilir:
- **Whisper Modeli:** Varsayılan `large-v3-turbo` (yüksek hız ve Türkçe doğruluk).
- **Segment Süresi:** Minimum 2.0 sn, maksimum 16.0 sn.
- **Kırpma Boyutu:** 96×96 piksel, 25 fps.
- **Dudak Senkronizasyonu (Lip-Sync):** Ağız açıklığı standart sapması, hareket eşikleri ve SyncNet skor sınırları.
