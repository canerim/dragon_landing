# KAIROS — Türkçe Özet

**RSNA Knee Abnormality Detection (Kaggle 2026) için çok modlu, geometri-farkında,
uyarlanabilir-hesaplı sistem.**

Bu belge, elinizdeki 29 sayfalık strateji dokümanına göre **neyin değiştiğini** ve
**neden değiştiğini** anlatır. Tam yöntem tarifi `docs/DESIGN.md` içinde (İngilizce,
kazanan yükümlülüğü olan açık kaynak teslimatı ve RSNA sunumu için).

---

## 1. Mevcut dokümanınız ne yapıyordu, bu ne ekliyor

Sizin dokümanınız sağlam bir yarışma planı: hasta bazlı fold, 2.5D ConvNeXt/Swin,
attention pooling, ASL, çeşitli ensemble, per-label ağırlık, distile öğrenci.
Bu plan iyi bir derece getirir. Aşağıdaki yedi şey onun **üstüne** gelen ve
her biri OOF üzerinde ölçülebilir olan farklar — beklenen etki sırasına göre:

| # | Fark | Sizin dokümanınızda | Burada |
|---|------|--------------------|--------|
| 1 | Metrik-doğrudan optimizasyon | "AUC ranking loss, düşük ağırlık, geç" (öneri) | AUC min-max margin **kanıtlanmış özdeşlikle**, two-way partial AUC, nadir etiketler için momentum kuyruk; PESG min-max optimizer |
| 2 | Fiziksel geometri | Slice sıralaması için doğru formül | Sıralama + **milimetre cinsinden** Fourier pozisyon kodu, metrik rotary attention, metrik ALiBi, SSM'de fiziksel Δ, fiziksel spacing'e resample |
| 3 | Label-specific yapı | 12 label query | 12 query **+ top-2 MoE (mekanizma taksonomisinden yönlendirilmiş) + Poincaré ontoloji önselı + entailment cone** |
| 4 | Rapor denetimi | Contrastive + weak label + KD (kavramsal) | **Kutu olmadan** unbalanced optimal transport ile phrase→slice grounding, klinik-benzerlik soft target, **shortcut regularizer + bloklayıcı denetim** |
| 5 | Domain shift | "Alt grup analizi yap" | Group-DRO **+ standart-hata shrinkage + label standardizasyonu**, χ²-DRO, IRMv1; site AUC farkı birinci sınıf metrik |
| 6 | Adaptif hesap | Coarse-to-fine top-k fikri | **Eğitilmiş** Gumbel top-k + güven/belirsizlik kapısı + bütçe kaybı + ölçülmüş Pareto eğrisi + **kapalı çevrim runtime governor** |
| 7 | İstatistik | Bootstrap öner | **DeLong eşleşmiş test**, hasta-kümeli bootstrap, **iç içe (nested) ensemble doğrulaması**, 10'lu shortcut denetim paketi (5'i bloklayıcı) |

---

## 2. En kritik üç fark (detay)

### 2.1 AUC'yi gerçekten optimize etmek

Squared-hinge AUC surrogate'ının $O(n^+n^-)$ çift toplamı, yardımcı değişkenlerle
**örnek-başına** ifadeye çöker:

$$
\min_{\theta,a,b}\max_{\alpha\ge0}\;
(1-p)\mathbb E[(h-a)^2\mathbb 1_{y=1}]
+ p\,\mathbb E[(h-b)^2\mathbb 1_{y=0}]
+ 2\alpha\big(p(1-p)m + \mathbb E[p h\mathbb 1_{y=0}-(1-p)h\mathbb 1_{y=1}]\big)
- p(1-p)\alpha^2
$$

Eyer noktasında bu tam olarak $p(1-p)\,\mathbb E[(m-h(x^+)+h(x^-))^2]$'ye eşittir.

**Dikkat:** beklentiler *koşullu ortalama değil*, sınıf göstergeli **tüm-batch
ortalaması** olmak zorunda. Koşullu ortalama kullanırsanız iki varyans terimi
$p(1-p)$ yerine $(1-p)$ ve $p$ ile ağırlıklanır ve özdeşlik ~6 kat sapar. Bu hatayı
geliştirme sırasında testler yakaladı; `tests/test_torch_components.py` içinde
özdeşlik uçtan uca doğrulanıyor.

$A_1$ ve $A_2$ birer **varyans** terimidir: pozitif skorları birbirine, negatif
skorları birbirine çeker, ama hiçbirini 0 veya 1'e itmez. AUC-M'in BCE doyduktan
çok sonra bile sıralamayı iyileştirmesinin sebebi budur.

**Nadir etiket sorunu:** %0.8 prevalanslı bir etikette 32 çalışmalık batch'te
pozitif dörtte bir ihtimalle bulunur; diğer üç batch'te o etiket için **hiç gradyan
yoktur**. Sampler'ı zorlamak birlikte görülen etiketleri de bozar (Fracture'ı
oversample etmek Contusion'ı da oversample eder). Çözüm: son 512 skoru etiket ve
sınıf başına tutan detached momentum kuyruğu — adım başına etkin çift sayısı
$O(1)$'den $O(Q)$'ya çıkar.

### 2.2 Fiziksel geometri her yerde

`z / N` (indeks) yerine **milimetre**. Fourier pozisyon kodunun periyotları 240 mm
(tüm diz) ile 2 mm (menisküs kökü) arasında log-aralıklı. Bir MLP'nin ham skaler
koordinat üzerindeki düşük-frekans spektral yanlılığı "slice 14, slice 15'ten
farklıdır" ifadesini temsil **edemez** — ki diz patolojisi tam olarak bu çözünürlükte
yaşar.

Aynı ikame attention'a da uygulanıyor: RoPE token indeksi yerine metrik koordinatla
sürülüyor, böylece attention **fiziksel uzayda** öteleme-eşdeğişken oluyor ve
stack ortasındaki 6 mm'lik boşluk gizlenmek yerine dürüstçe temsil ediliyor.

Selective SSM'de ayrıklaştırma adımı $\Delta_i$, **gerçek slice aralığıyla**
ölçekleniyor. SSM'in attention'dan kesinlikle daha doğal olduğu tek yer burası:
sürekli-zaman formülasyonu zaten düzensiz aralıklı örneklenmiş bir sinyali modeller.

Resample **fiziksel spacing'e** (0.70 mm coarse, 0.35 mm fine), sabit piksel
ızgarasına değil. 3 mm'lik menisküs kök yırtığı 16 merkezin hepsinde aynı sayıda
piksel kaplar.

### 2.3 Rapor: ayrıcalıklı bilgi, test-time kısayolu değil

**Kutu yok**, o yüzden doğal matematiksel nesne bir **transport planı**:
cümle gömüleri ile slice gömüleri arasında, maliyet $C_{ij}=1-\langle u_i,v_j\rangle$.

İki sapma zorunlu:

- **Unbalanced (dengesiz) OT.** Her cümlenin görsel karşılığı yok ("klinik öykü:
  ağrı"); $T\mathbf 1=\mu$ zorlaması **sahte hizalama üretir**. KL marjinal
  gevşetmesi ($\tau$) Sinkhorn güncellemesini sönümler ve eşleşme olmayan yerde
  kütlenin yok edilmesine izin verir. $\tau=0.5,\varepsilon=0.05$'te cümlelerin
  yaklaşık üçte biri taşınmadan kalıyor — bu, bir diz raporunun öykü/teknik/
  karşılaştırma kısmının oranıyla örtüşüyor.
- **Log-domain.** $e^{-C/\varepsilon}$ fp16'da anında taşar.

Gradyan **zarf teoremi** ile: optimumda $\partial\mathrm{OT}/\partial C=T^\star$,
yani plan detach edilir ve sadece $\langle T^\star,C\rangle$ üzerinden geri yayılır.
İterasyonları açmaktan hem ucuz hem çok daha kararlı.

**Kısayol koruması** — bu yönün tamamını batırabilecek arıza modu, "çok modlu
öğretmen"in gizlice rapor-sınıflandırıcısına dönüşmesidir. Kendi validasyon
AUC'sinde görünmez, ama image-only öğrenci için değersizdir. Ceza:

$$
\mathbb E[\mathrm{ReLU}(\mathcal C(z^{\text{karışık}})-\mathcal C(z^{\text{doğru}})+\delta)]
+\eta\,\mathbb E[\mathrm{KL}(\sigma(z^{\text{karışık}})\|\sigma(z^{\text{görüntü}}))]
$$

Her epoch loglanan `report_reliance` skaleri, çok modlu öğretmenin gerçek olup
olmadığını söyleyen **tek sayıdır**.

---

## 3. Çok dilli rapor ayrıştırma: üç tuzak

RadGraph / CheXbert / NegBio hem İngilizce hem göğüs-özgüdür. **Yöntemi** taşıyın,
**modeli** değil. Sözlük 12 dilde 12 etiketi kapsıyor. Naif bir port'un kaçırdığı üç şey:

1. **Türkçe büyük/küçük harf.** Noktasız `ı` / noktalı `i` çifti yüzünden Python'un
   varsayılan `.lower()`'ı `MENİSKÜS`'ü birleşen noktalı `meni̇sküs`'e çevirir ve
   `menisküs` ile eşleşmez. Çözüm: case-fold **sonrası** NFKC ve eşleşme anahtarından
   birleşen işaretleri düşürmek.
2. **Sonda gelen olumsuzlama.** NegEx sol-bağlam kapsam kuralı uygular. Türkçe
   (`… izlenmemektedir`) ve Japonca (`… 認めない`) kavramdan **sonra** olumsuzlar;
   sadece-sol kural bu iki dili sistematik olarak pozitif etiketler.
3. **Rapor yapısı.** Raporlar düzyazı değil; madde listesi ve noktalı virgülle
   zincirlenmiş cümlelerdir. Hazır bir cümle bölücü tüm findings bölümünü tek
   "cümle" yapar, olumsuzlama kapsamını ve ondan türeyen her weak label'ı yok eder.

Ayrıca kompartman çözücü var: "medial menisküs arka boynuz" ifadesi
`Lateral Meniscus` weak label'ı **üretemez**.

Çakışan bahisler POZİTİF > BELİRSİZ > NEGATİF sırasıyla çözülür. Gerekçe: olumsuzlar
genellikle şablon kontrol listesidir ("ACL intakt, PCL intakt, …"), yüksek hacimli ve
düşük bilgilidir; tek bir açık pozitifi geçmesine izin vermek, %99 özgüllük ve %40
duyarlılıkta bir weak labeller üretir.

**Sözlük bir başlangıçtır, teslimat değil.** `scripts/02_parse_reports.py --audit`
dil başına en sık eşleşmeyen cümleleri döker; sözlük korpusun gerçekte ne dediğinden
büyütülür.

---

## 4. Fold: kısıtlı optimizasyon problemi olarak

Düz stratified k-fold aynı anda üç konuda yetersiz: hasta gruplamasına saygı
gösteremez, multilabel marjinalleri kötü ele alır, site/dil dengesini hiç görmez.
Biz bölmeyi **optimize ediyoruz**:

$$
J(\pi)=
\lambda_{\text{mar}}\!\sum_{k,l}\!\frac{w_l(p_{kl}-p_l)^2}{p_l(1-p_l)+\varepsilon}
+\lambda_{\text{co}}\!\sum_k\!\lVert C_k-\bar C\rVert_F^2
+\lambda_{\text{cov}}\!\sum_{k,c}\!\mathrm{KL}(q_{kc}\|q_c)
+\lambda_{\text{sz}}\!\sum_k(n_k-\bar n)^2
$$

Paydadaki Bernoulli varyansı ilk terimi ki-kare benzeri bir istatistiğe çevirir;
böylece prevalansı bir mertebe farklı etiketler karşılaştırılabilir olur.
İki aşama: grup-seviyesi iterative stratification (tohum) → simulated annealing
(Metropolis kabul, artımlı delta).

Sentetik kohortta ölçülen etki: **en nadir etiketin fold-başı prevalans sapması
%100'ün üzerinden %40'ın altına** düşüyor. Bu, 0.004'lük bir AUC etkisini
çözebilmek ile çözememek arasındaki farktır.

**Fold artefaktı değişmezdir.** `fold_hash` her checkpoint manifestinde saklanır ve
`Trainer.load` uyuşmazlıkta yüklemeyi **reddeder**. Bölme sürümleri arasında sessizce
ensemble yapmak, sonradan kimsenin yeniden kuramayacağı miktarda iyimser bir OOF
skoru üretir.

---

## 5. Hangi sayıya güveneceksiniz

| Metrik | Neden |
|--------|-------|
| Macro AUC + hasta-kümeli bootstrap GA | Yarışma metriği ve gerçek belirsizliği |
| **Worst-label AUC** | Macro ortalama bunu belirleyici yapar |
| Fold standart sapması | Bölme hassasiyeti |
| **Site AUC farkı** (en iyi − en kötü) | Group-DRO'nun küçültmeye çalıştığı sayı; private sürprizi öngörür |
| DeLong **eşleşmiş** p-değeri | İki modeli aynı çalışmalarda karşılaştırırken kovaryans terimi olmadan gerçek iyileşmeler kaybolur |
| Nested (LOFO) ensemble kazancı | Per-label ağırlığın gerçek mi gürültü mü olduğu |
| Ölçülmüş notebook runtime | Efficiency track ve 9 saat sınırı |

**Per-label ensemble ağırlığı kuralı:** `06_ensemble.py`, nested ağırlıklı macro,
nested uniform macro'yu bir bootstrap standart hatasından fazla geçmedikçe
**uniform ortalamayı gönderir**. Geçersiz kılmak için `--force-weighted` gerekir.
12 etiket × M model = $12(M-1)$ serbest parametre; en nadir etikette etkin örneklem
belki 40 pozitif. Bu, tıbbi görüntü yarışmalarında en yaygın geç-dönem kendine
açılan yaradır.

---

## 6. Denetimler (5'i bloklayıcı)

Her biri, OOF'yi sağlıklı ya da **daha iyi** gösterirken private skoru bozmanın
bilinen bir yolunu yakalar:

| Denetim | Başarısızlık ne demek | Bloklayıcı |
|---------|----------------------|:---:|
| `shuffled_label` | Tahmin/etiket satırları hizasız — pipeline'ın en yıkıcı ve en kolay kaçırılan hatası | ✅ |
| `shuffled_report` | Çok modlu model aslında text-only | ✅ |
| `duplicate_hash` | Aynı çalışma iki fold'da | ✅ |
| `embedding_neighbour` | Yakın-kopya çalışmalar fold sınırını aşıyor | ✅ |
| fold-hash kontrolü | OOF matrisi izlenemiyor | ✅ |
| `metadata_only` | Site/protokol etiketi tahmin ediyor | ⚠️ |
| `fold_prevalence` | Bölme, aradığınız etkiyi ölçemeyecek kadar dengesiz | ⚠️ |
| `prediction_site_gap` | Model gizli bir site sınıflandırıcısı | ⚠️ |

`scripts/05_oof_eval.py` bloklayıcı bir hatada **sıfırdan farklı çıkar**. Sadece
uyaran bir denetim, son günün gecesinde göz ardı edilen denetimdir.

---

## 7. Efficiency track

Efficiency skoru runtime'ı **lineer**, doğruluğu normalize edilmiş açık üzerinden
alır. Sonuç: frontier yakınında az AUC'yi çok runtime'a takas etmek neredeyse her
zaman doğrudur — ve "yakın"ın nerede olduğunu bilmenin tek yolu ölçmektir.

Toplam süre $t_{\text{decode}}+t_{\text{preproc}}+t_{\text{load}}+t_{\text{infer}}+t_{\text{post}}$,
yani hızlı GPU modeli seri DICOM decode'unda kaybedebilir. Getiri sırası: önceden
hazır manifest → batched/GPU decode (HTJ2K'da nvJPEG2000 ~5×) → volume cache →
coarse-to-fine top-k → dinamik çözünürlük → sequence skipping → BF16 → (ancak
profiling model darboğazını doğruladıktan sonra) structured pruning ve INT8.

**Governor:** Kaggle 9 saat verir, test seti boyutu bilinmez (public'in 3 katı
olabilir). Sabit politika ya bütçeyi çöpe atar ya taşar. Oransal denetleyici
gerçekleşen maliyeti izler ve projeksiyon %12 rezervli bütçeye otursun diye
eskalasyonu ayarlar. **CSV üretmeyen submission sıfır alır**, o yüzden governor
taşmak yerine coarse-only'ye düşer.

**Öğrenci ranking kaybı görmez.** Öğretmenin logitleri sıralamayı zaten kodluyor;
üstüne AUC-M eklemek öğrenciyi öğretmen-uyumunu kendi gürültülü sıralama tahminiyle
takas etmeye zorlar — 150'den az pozitifi olan her etikette ölçülebilir biçimde kötü.

Öğretmen logitleri **cross-fitted** olmalı: çalışmanın OOF öğretmen tahmininden,
o çalışma üzerinde eğitilmiş öğretmenden asla değil.

---

## 8. Ensemble ve hazır ağırlıklar

Çeşitlilik **ölçülür**, varsayılmaz: üyeler OOF hata korelasyonuna göre seçilir;
artıkları mevcut bir üyeyle 0.9'un üzerinde korele olan üye, tek başına skoru ne
olursa olsun düşürülür. Tek bir backbone'un on seed'i ensemble değildir.

| # | Backbone | Aggregator | Ayırt edici inductive bias |
|---|----------|-----------|---------------------------|
| 1 | ConvNeXt-S (IN-22k) | transformer | modern CNN lokallik |
| 2 | Swin-S | transformer | hiyerarşik pencere attention |
| 3 | DINOv2/v3 ViT-B + LoRA | SSM | SSL öznitelikleri, lineer-zamanlı slice |
| 4 | MedSigLIP-400M (donuk + plugin) | transformer | tıbbi VLP öznitelikleri |
| 5 | 3D ResNet / Video-Swin | native 3D | gerçek volumetrik bağlam |
| 6 | nadir-etiket uzmanı | transformer | Fracture/Contusion duyarlılığı |

**Aday hazır ağırlıklar** (hepsi Kaggle Dataset olarak çevrimdışı paketlenebilir,
hepsi aynı fold ve aynı bütçede ImageNet init'e karşı ablate edilmeli):
`timm` ConvNeXt/Swin/EfficientNetV2; DINOv2 ve **DINOv3** (Gram-anchored patch
öznitelikleri, yayımlanmış ConvNeXt distilasyonları); **MedSigLIP-400M** (tıbbi
ayarlı SigLIP görü kulesi, 448 px, açık ağırlık); **OrthoFoundation** (DINOv3
omurga, 1.2M etiketsiz diz X-ray/MRI — ağırlıklar zamanında yayımlanırsa);
**Triad** (3D MRI temel modeli, 131k volüm); 3D dal için MedicalNet / Models
Genesis; radyoloji-alanı 2D init için RadImageNet; **sadece lokalizatör** olarak
TotalSegmentator MRI veya nnU-Net diz modeli — asla sınıflandırıcı olarak.

> **Kritik bulgu:** CVPR 2026 *"Revisiting 2D Foundation Models for Scalable 3D
> Medical Image Classification"* 12 volumetrik görevde şunu ölçüyor: doğru adapte
> edilmiş **2D temel modeller native 3D mimarileri geçiyor**, genel amaçlı SSL
> omurgaları tıbbi-özgü olanlarla eşleşiyor, ve **adaptasyon mekanizması omurga
> seçiminden daha belirleyici**. Bu, son RSNA-Kaggle sonuçlarıyla tutarlı: 2022
> servikal omurga, 2023 abdominal travma, 2024 lomber omurga — hepsi 2.5D encoder
> + sekans başlığı tasarımlarıyla kazanıldı. Bu yüzden gerçek 3D dal **yalnızca
> ensemble çeşitliliği için** tutuluyor.

---

## 9. Sıra (11 hafta)

Sıralama önem sırasına göre değil, **bir sonraki şeyin anlamlı olması için neyin
doğru olması gerektiğine** göre:

| Hafta | Teslimat | Geçme kriteri |
|-------|----------|---------------|
| 1 | DICOM manifest + QC + geometrik sıralama | <%0.5 decode hatası; 50 seri elle doğrulanmış |
| 1–2 | Fold artefaktı + hash + 2.5D baseline | Nadir etiket sapması <%40; bloklayıcı denetimler geçiyor |
| 2 | **Ölçülmüş Kaggle runtime** (dummy submission) | Uçtan uca <9 saat, pay bırakarak |
| 3 | Rapor parser + text-only baseline | Dil başına eşleşmeyen cümle denetimi incelendi |
| 3–4 | Label query + cross-sequence fusion | Macro-AUC ↑ **ve** en az 4 zor etiket ↑ |
| 4–5 | Image–report pretraining | Aynı bütçede image-only OOF ↑; `report_reliance` > 0 |
| 5–6 | Coarse-to-fine adaptif | Aynı AUC'de daha düşük runtime ya da tersi |
| 6–7 | 3D/Video-Swin çeşitlilik dalı | 1–3 ile OOF artık korelasyonu <0.9 |
| 7–8 | Ranking fine-tune | Macro-AUC ↑, worst-label AUC ↓ değil |
| 8–9 | Robustness aşaması | Site AUC farkı ↓, macro-AUC ↓ değil |
| 9–10 | Ensemble + nested kontrol | Nested ağırlıklı > nested uniform + 1 SE, yoksa uniform gönder |
| 10 | Efficiency öğrenci | Öğretmen macro-AUC'sinin ≥%90'ı, ≤%25 runtime'da |
| 11 | Notebook sertleştirme, freeze | Sınır içinde üç temiz commit koşusu |

**Sıfırıncı öncelik**, hepsinden önce: kusursuz hasta bazlı beş fold, güvenilir DICOM
pipeline'ı, OOF tahminleri ve **ölçülmüş** Kaggle inference süresi. Bu dördü var
olmadan mimariye harcanan her saat, güvenemeyeceğiniz bir sayıya harcanmıştır.

---

## 10. Nasıl çalıştırılır

```bash
pip install -e ".[train,dev]"
pytest -q                          # 196 test
python scripts/99_smoke_test.py    # 10 aşamalı uçtan uca koşu, ~20 sn, sadece CPU

# tüm eğitim yolunun kuru koşusu: veri yok, GPU yok, ~1 dk
python scripts/04_train.py --synthetic --budget small --epochs-cap 2 \
    --no-pretrained --device cpu --disable kd,ot_ground,weak_label
```

Sıra: `00_build_manifest` → `01_make_folds` → `02_parse_reports --audit` →
`04_train` → `05_oof_eval` → `06_ensemble`. Hepsinden önce
`notebooks/kaggle_baseline.py`'ı Kaggle'a at — ağırlık gerektirmiyor, ~0.5 alıyor,
ve karşılığında veri düzenini, DICOM tag sayımını ve I/O'nun 9 saatin ne kadarını
yediğini söylüyor.

**Bir güvenlik özelliği:** curriculum hesaplanamayacak bir terim planlarsa
(`kd` var ama teacher logit yok gibi) eğitim **başlamıyor**, eksik alanı
söyleyerek duruyor. Sessizce atlamak, daha küçük bir hedefi optimize edip loss
eğrisini sağlıklı gösterirdi — ve tek belirti, üç GPU-günü sonra ablasyonla
çelişen bir OOF skoru olurdu. Bilerek kapattığın terimleri `--disable` ile
adlandırman gerekiyor ve run manifest'ine yazılıyorlar.

Smoke test gerçek kod yolunu sürüyor — fold → collation → curriculum altında model
ileri/geri → OOF değerlendirme → denetim paketi → ensembling → kalibrasyon →
conformal → submission. Stub yok.

---

## 11. Testlerin yakaladığı dört gerçek hata

Geliştirme sırasında testler dört gerçek hata buldu; her biri düzeltildiği yerde
kodda belgeli:

| # | Hata | Neden önemliydi |
|---|------|-----------------|
| 1 | Kanonik yön çevirmesi normali negatifliyor ama eski projeksiyonu kullanıyordu | Fiziksel koordinat azalan kalıyordu |
| 2 | AUC-M'de koşullu ortalama | Min-max özdeşliği ~6 kat sapıyordu |
| 3 | `np.nan_to_num` `+inf`'i 1.8e308'e çeviriyor | Otsu'nun argmax'ı hep son bin'i seçiyor, maske sessizce fallback'e düşüyordu |
| 4 | `\b` alt çizgide bölmüyor | `sag_pdw_fs_tse` yağ-baskısız sanılıyordu |
| 5 | `build_submission` kendi sabit fallback'ini doğruluyordu | Dosya varlığını **garantileyen** güvenlik ağı exception atıyordu |
| 6 | SNGP kovaryansı iki head çağrısı arasında in-place tazeliyordu | `backward()` version-counter hatasıyla düşüyordu |
| 7 | `ConfidenceGate` boolean eşikleri "öğrenilebilir" işaretliyordu | Asla gradyan alamayacak parametreler |
| 8 | `KneeOntology` `slots=True` altında tanımsız attribute atıyordu | Construct etmek exception atıyordu; hiç instantiate edilmemişti |
| 9 | `normalise_text` casefold'un birleşen noktasını temizlediğini iddia ediyordu | Temizlemiyordu |
| 10 | LID kana'dan önce CJK'yı kontrol ediyordu | Her Japonca rapor Çince etiketleniyordu |

Bunların hiçbiri "şekil testi" ile yakalanamazdı. Test paketi şekil değil,
**sayısal özellik** sabitler. En belirleyicisi
`test_model_can_overfit_a_tiny_dataset`: birleştirilmiş graf 12 çalışmayı
macro-AUC 1.000'e sürüyor — gradyan yolunun loss'tan SNGP head, MoE router,
ontoloji önselı, cross-sequence fusion, label query, aggregator, FiLM ve
backbone boyunca sağlam olduğunun kanıtı.
