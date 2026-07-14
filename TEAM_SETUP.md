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

### 3.2 Kendi videolarını işle

Bulduğun YouTube linklerini `sources_no_voiceover.txt` (veya voiceover'lı olanları
`sources_voiceover.txt`) dosyasına, satır başına bir link olacak şekilde yaz. Sonra:

```bash
yt2avsr process-both-sources --config configs/default.yaml
# veya tek liste:
yt2avsr process-sources sources_no_voiceover.txt --config configs/default.yaml
```

Çıktı lokalde `data/clips/...` altında oluşur ve `data/manifests/accepted.csv`
kullanılabilir klipleri listeler.

### 3.3 Verini buluta gönder

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
