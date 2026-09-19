# Ekip Kurulum Kılavuzu (GitHub + Bulut Veri)

Bu kılavuz iki şeyi anlatır:

1. **Kodu GitHub'a** yüklemek (sen bir kez yaparsın).
2. **Üretilen eğitim verisini** ortak bir bulut deposunda (Hugging Face) toplamak,
   arkadaşların oraya veri göndermesi ve Auto-AVSR eğitiminin oradan veri çekmesi.

Kod GitHub'a, **veri** Hugging Face'e gider. Videolar/ses dosyaları büyük olduğu için
GitHub'a konmaz (zaten `.gitignore` bunları engelliyor).

---

## Bölüm 1 — Kodu GitHub'a yükle (SADECE SEN, bir kez)

### 1.1 Yarım kalan git klasörünü temizle

Kod klasöründe daha önce yarım bir `.git` oluştu. Kendi **Terminal'inde** (macOS)
şu klasöre gir ve sıfırla:

```bash
cd ~/Downloads/youtube-to-autoavsr-v0.7-fixed
rm -rf .git
```

### 1.2 GitHub'da boş bir private repo aç

github.com → sağ üst **+** → **New repository**:

- Repository name: `youtube-to-autoavsr` (istediğin ad)
- Görünürlük: **Private** ✅
- **README/gitignore/license EKLEME** (bizde zaten var) → **Create repository**

Açılan sayfadaki repo URL'sini kopyala, örn:
`https://github.com/KULLANICI_ADIN/youtube-to-autoavsr.git`

### 1.3 Kodu yükle

Terminal'de (klasörün içindeyken):

```bash
git init
git add -A
git commit -m "İlk sürüm: yt2avsr v0.7 + bulut senkronizasyonu"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADIN/youtube-to-autoavsr.git
git push -u origin main
```

> `git commit` ilk kez isim/e-posta isterse:
> ```bash
> git config --global user.name "Adın"
> git config --global user.email "mail@örnek.com"
> ```

### 1.4 Arkadaşlarını davet et

GitHub'da repo → **Settings → Collaborators → Add people** → arkadaşlarının
GitHub kullanıcı adlarını ekle. (Private repo olduğu için davet şart.)

---

## Bölüm 2 — Ortak bulut deposunu kur (SADECE SEN, bir kez)

Veri için **Hugging Face** kullanıyoruz: her yerden (Colab, üniversite GPU'su,
bulut GPU) aynı komutla erişilir, private repo destekler, büyük dosyaları otomatik
taşır, ücretsiz katmanı geniştir.

### 2.1 Hesap ve token

1. huggingface.co → ücretsiz hesap aç.
2. **Settings → Access Tokens → New token** → tür **Write** → oluştur, kopyala.

### 2.2 Ortak dataset repo'sunu aç

huggingface.co → sağ üst profil → **New Dataset**:

- Owner: kendi kullanıcı adın ya da bir ekip/organizasyon
- Dataset name: `avsr-tr-dataset`
- **Private** ✅ → **Create dataset**

Repo kimliği şu biçimde olur: `KULLANICI_ADIN/avsr-tr-dataset`

### 2.3 Repo kimliğini config'e yaz ve GitHub'a gönder

`configs/default.yaml` dosyasında:

```yaml
cloud:
  repo_id: "KULLANICI_ADIN/avsr-tr-dataset"
  private: true
```

Sonra bu değişikliği GitHub'a gönder ki herkes aynı repoyu kullansın:

```bash
git add configs/default.yaml
git commit -m "Ortak HF dataset repo'sunu ayarla"
git push
```

### 2.4 Arkadaşlarına HF erişimi ver

HF'de dataset → **Settings → (üyeler / collaborators)** → arkadaşlarının HF
kullanıcı adlarını **write** yetkisiyle ekle. (Bir organizasyon açıp herkesi oraya
davet etmek en temiz yöntemdir.)

---

## Bölüm 3 — Her arkadaşın yapacağı (herkes kendi bilgisayarında)

### 3.1 Kurulum (bir kez)

```bash
brew install ffmpeg git
git clone https://github.com/KULLANICI_ADIN/youtube-to-autoavsr.git
cd youtube-to-autoavsr
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
yt2avsr setup-external --config configs/default.yaml
huggingface-cli login       # HF token'ını yapıştır (bir kez)
```

### 3.2 Önceki işlenmiş videoları senkronize et (Mükerrer önleme)

Başkalarının daha önce işleyip Hugging Face'e attığı videoları çekmek için:

```bash
yt2avsr sync-processed --config configs/default.yaml
```

Bu sayede `processed_sources.txt` güncellenir ve diğer ekip üyelerinin işlediği videolar senin listende olsa bile otomatik atlanır.

### 3.3 Videoları ekle ve işle (veya ekiple paylaşımlı çalış)

Linkleri dosyalara elle yapıştırmak Git çakışmalarına (conflict) ve mükerrer videolara yol açabilir. Bunun yerine her zaman **`ytavsr add`** (veya Windows'ta `yt2avsr add`) komutunu kullanın.

Bu komut otomatik olarak:
1. Ekip arkadaşlarının eklediği son linkleri çeker (`git pull`).
2. Mükerrer veya önceden işlenmiş videoları filtreler.
3. Dosyaya ekleyip GitHub'a otomatik yükler (`git push`).

```bash
# Tek video eklemek:
ytavsr add "https://www.youtube.com/watch?v=VIDEO_ID"

# Birden fazla videoyu aynı anda eklemek:
ytavsr add "URL1" "URL2" "URL3"

# Dosyadan topluca eklemek:
ytavsr add linkler.txt

# Dış ses / dublaj içeren videolar için:
ytavsr add "URL" --voiceover
```

**3 kişi aynı anda çalışıyorsanız, çakışmayı önlemek için `--shard` kullanın:**
- 1. Bilgisayar: `ytavsr --shard 0/3`
- 2. Bilgisayar: `ytavsr --shard 1/3`
- 3. Bilgisayar: `ytavsr --shard 2/3`

Tek başına çalışıyorsan:
```bash
ytavsr
# veya:
yt2avsr process-both-sources --config configs/default.yaml
```

Çıktı lokalde `data/clips/...` altında oluşur ve `data/manifests/accepted.csv`
kullanılabilir klipleri listeler.

### 3.4 Verini buluta gönder

```bash
yt2avsr push-data --config configs/default.yaml
```

Bu, senin `accepted` + `review` kliplerini ortak repoda **kendi adına ait** bir
alt klasöre yükler (`data/<senin-hf-kullanıcı-adın>/...`). Kimse kimsenin verisini
ezmez.

- Sadece kesin kabul edilenleri yükle: `--include accepted`
- Alt klasör adını elle ver: `--contributor ibrahim`

---

## Bölüm 4 — Eğitim makinesinde veriyi topla

Auto-AVSR'ı nerede eğiteceksen (Colab, cluster, bulut GPU), orada:

```bash
huggingface-cli login
yt2avsr pull-data --config configs/default.yaml --dest data_cloud
```

Herkesin verisi tek bir kök altına iner:

```text
data_cloud/data/
├── ibrahim/    clips/... + manifests/...
├── arkadas2/   clips/... + manifests/...
└── arkadas3/   clips/... + manifests/...
```

Auto-AVSR eğitim scriptini bu klasördeki `mouth.mp4` + `transcript.txt` çiftlerine
yönlendir. Her kişinin `manifests/accepted.csv` dosyasındaki yollar, kendi klip
klasörlerine görecelidir.

> Yeni veri eklendikçe eğitim makinesinde tekrar `pull-data` çalıştırman yeterli;
> Hugging Face yalnızca değişenleri indirir.

---

## Bölüm 5 — Modal ile Bulutta GPU İşleme (RetinaFace + 1080p)

Mac veya dizüstü bilgisayarlarda CPU ile **RetinaFace** çalıştırmak çok yavaş kalabilir. Resmi Auto-AVSR kalitesinde 1080p video ve RetinaFace dudak takibini en hızlı şekilde yapmak için **Modal** (modal.com) entegrasyonunu kullanabilirsiniz.

> Modal her kullanıcıya **aylık 30$ ücretsiz GPU kredisi** verir. Ekipteki herkes kendi kişisel Modal hesabını bağlayarak kendi GPU kotasıyla işleme yapabilir.

### 5.1 ÖNEMLİ ÖN KOŞUL: Hugging Face Girişi Yapın
Modal'da işlenen kliplerin doğrudan ortak depoya yüklenebilmesi için bilgisayarınızda Hugging Face oturumunuzun açık olması **şarttır**:

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) adresine gidin.
2. **Write** yetkili bir token oluşturun ve kopyalayın.
3. Terminalde çalıştırıp yapıştırın:
   ```bash
   huggingface-cli login
   ```
*(Bu işlemi bir kez yaptıktan sonra Modal token'ınızı yerelden otomatik okur).*

### 5.2 Modal Kurulumu (Kişi Başı Sadece 1 Kez)

1. [modal.com](https://modal.com) adresine gidin ve GitHub hesabınızla ücretsiz kaydolun.
2. Terminalinizde Modal paketini kurun:
   ```bash
   pip install modal
   ```
3. Hesabınızı terminale bağlayın:
   ```bash
   modal setup
   ```
   *(Açılan tarayıcı penceresinde onay vermeniz yeterlidir).*

### 5.3 Videoları Modal GPU'sunda İşleme

Artık tüm indirme, Whisper transkripsiyonu, 1080p RetinaFace ağız kesimi ve Hugging Face'e yükleme işlemleri bulutta NVIDIA T4 GPU üzerinde gerçekleşir:

```bash
# Kaynak dosyalarındaki (sources_*.txt) tüm bekleyen videoları işle:
ytavsr modal

# veya doğrudan Modal CLI ile:
modal run modal_app.py
```

#### Ekiple Çakışmasız Paralel Çalışma (Sharding):
Aynı anda 3 kişi çalışıyorsanız listeyi 3 parçaya bölerek çalıştırın:
- 1. Kişi: `ytavsr modal --shard 0/3`
- 2. Kişi: `ytavsr modal --shard 1/3`
- 3. Kişi: `ytavsr modal --shard 2/3`

#### Tek Video veya Özel Seçenekler:
```bash
# Tek bir videoyu bulut GPU'sunda test etmek:
ytavsr modal --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Dış ses / dublaj içeren bir video için:
ytavsr modal --url "https://www.youtube.com/watch?v=VIDEO_ID" --voiceover

# Klipleri sadece Hugging Face'e atmakla kalmayıp yerel bilgisayarınıza da indirmek için:
ytavsr modal --download-local
```

### 5.4 İşlem Sonrası Ne Olur?
1. Modal'daki GPU container'ı işlenen klipleri doğrudan ortak Hugging Face dataset'ine (`iboRotti/avsr-tr-dataset`) yükler (`data/<kullanıcı-adınız>/...`).
2. Hugging Face'teki ve yerelinizdeki `processed_sources.txt` listesi otomatik güncellenir.
3. Diğer ekip arkadaşlarınız işlem başlattığında bu videolar otomatik atlanır (mükerrer işleme engellenir).

---

## Sık sorulanlar

**Neden GitHub'a veri koymuyoruz?** GitHub kod içindir; büyük video/ses için değil.
`.gitignore` `data/`, `data_cloud/`, `external/` ve token dosyalarını dışlar.

**Token'ım GitHub'a sızar mı?** Hayır. `huggingface-cli login` token'ı işletim
sistemi profiline kaydeder, repoya değil. Ayrıca `.env`/`*.token` dosyaları
`.gitignore`'da.

**Veri çok büyürse?** Hugging Face büyük dosyaları otomatik (LFS ile) taşır.
Ücretsiz katman yetmezse HF'nin ücretli depolama planına geçebilir ya da
`configs/default.yaml`'da yalnızca `accepted` yükleyerek yer kazanabilirsin.

**Başka bir buluta geçmek istersek?** Kod bulut sağlayıcısından bağımsız; sadece
`src/yt2avsr/cloud.py` içindeki push/pull fonksiyonlarını (ör. S3/GCS) değiştirmek
yeterli. Geri kalan iş akışı aynı kalır.
