# (EN) This project belongs to an 18-year-old software developer. 

# (TR) Bu proje 18 yaşındaki yazılımcıya aittir.

EvoNet: Neuroevolution for Sorting Task - Project Evolution / Sıralama Görevi için Neuroevolution - Proje Gelişimi
(EN) This repository documents the evolution of EvoNet, a project exploring the use of neuroevolution to automatically design neural network architectures capable of sorting numerical sequences. It started as a learning exercise and evolved through different versions with increasing robustness and features.

(TR) Bu repo, sayısal dizileri sıralayabilen sinir ağı mimarilerini otomatik olarak tasarlamak için neuroevolution kullanımını araştıran bir proje olan EvoNet'in gelişimini belgelemektedir. Bir öğrenme alıştırması olarak başlamış ve artan sağlamlık ve özelliklerle farklı versiyonlara evrilmiştir.

Project Goal / Proje Amacı
(EN) The primary goal is to investigate how evolutionary algorithms can discover effective neural network topologies (specifically simple feed-forward networks using Keras Dense layers) for a fundamental task like sorting, without pre-defining the architecture. We track performance using Mean Squared Error (MSE) and Kendall's Tau rank correlation.

(TR) Temel amaç, evrimsel algoritmaların, mimarisi önceden tanımlanmadan, sıralama gibi temel bir görev için etkili sinir ağı topolojilerini (özellikle Keras Dense katmanlarını kullanan basit ileri beslemeli ağlar) nasıl keşfedebildiğini araştırmaktır. Performansı Ortalama Karesel Hata (MSE) ve Kendall's Tau sıralama korelasyonu kullanarak takip ediyoruz.

Versions / Versiyonlar
v1: The Baseline (evonet_optimizer.py)
(EN)

Goal: To establish a functional and robust pipeline for evolving random Keras topologies. The focus was on creating a configurable, reproducible experiment with proper logging, evaluation, and best model saving, applying a basic evolutionary strategy.
Key Features:
Command-line argument parsing (argparse) for configuration.
Detailed logging to file and console.
Random generation of Sequential models with varying hidden layers and neurons.
Evolutionary loop using Tournament Selection and Elitism.
Mutation primarily via Weight Perturbation (Gaussian noise). Activation mutation was experimental.
Final training of the best-evolved architecture using Keras fit with callbacks (EarlyStopping, ReduceLROnPlateau).
Evaluation using MSE and Kendall's Tau on a separate test set.
Outcome: This version successfully ran and demonstrated the feasibility of the approach. In a sample run (100 generations, pop size 80, seed 123, on CPU), the finally trained best model achieved:
Test MSE: ~38.6 (indicating reasonable accuracy in numerical values, significantly improved by final training).
Kendall's Tau: ~0.996 (indicating excellent performance in predicting the correct order of elements).
This showed that while perfect value prediction is hard, evolution could find architectures very good at the ranking aspect of sorting. The final training step proved crucial for performance.
(TR)

Amaç: Rastgele Keras topolojilerini evrimleştirmek için işlevsel ve sağlam bir iş akışı oluşturmak. Odak noktası, yapılandırılabilir, tekrarlanabilir, düzgün loglama, değerlendirme ve en iyi model kaydetme özelliklerine sahip, temel bir evrimsel strateji uygulayan bir deney yaratmaktı.
Ana Özellikler:
Yapılandırma için komut satırı argüman yönetimi (argparse).
Dosyaya ve konsola detaylı loglama.
Değişen gizli katman ve nöron sayılarına sahip Sequential modellerin rastgele üretimi.
Turnuva Seçilimi ve Elitizm kullanan evrim döngüsü.
Öncelikli olarak Ağırlık Bozulması (Gauss gürültüsü) ile mutasyon. Aktivasyon mutasyonu deneyseldi.
Evrimle bulunan en iyi mimarinin Keras fit ve callback'ler (EarlyStopping, ReduceLROnPlateau) ile son eğitimi.
Ayrı bir test seti üzerinde MSE ve Kendall's Tau ile değerlendirme.
Sonuç: Bu versiyon başarıyla çalıştı ve yaklaşımın fizibilitesini gösterdi. Örnek bir çalıştırmada (100 nesil, popülasyon 80, tohum 123, CPU üzerinde), son eğitilmiş en iyi model şu sonuçları elde etti:
Test MSE: ~38.6 (Sayısal değerlerde makul bir doğruluğu gösterir, son eğitimle önemli ölçüde iyileşmiştir).
Kendall's Tau: ~0.996 (Elemanların doğru sırasını tahmin etmede mükemmel performansı gösterir).
Bu sonuçlar, mükemmel değer tahmini zor olsa da, evrimin sıralamanın sıralama yönünde çok iyi mimariler bulabildiğini gösterdi. Son eğitim adımının performans için kritik olduğu kanıtlandı.
v2: Enhanced Exploration & Usability (evonet_optimizer_v3.py)
(EN)

Goal: To improve upon v1 by adding features for potentially better exploration of the solution space and enhanced usability for long-running experiments.
Key Improvements over v1:
Crossover: Introduced basic weight crossover (averaging/mixing) between architecturally compatible parent models. This aims to combine potentially good traits from different lineages.
Checkpointing: Implemented functionality to save the state of the evolution (population models, random states) at regular intervals (--checkpoint_interval). Allows resuming interrupted runs using --resume_from, crucial for long experiments, especially on CPUs.
Refined Structure: Code structure slightly improved for better maintainability and integration of new features.
Expected Outcome: While requiring experimental runs for validation, v2 is expected to be more robust for longer evolutionary processes. The addition of crossover may help escape local optima or converge faster in some cases. Checkpointing significantly improves the practicality of running extensive experiments. Performance metrics (MSE, Tau) would need to be evaluated after running v2.
(TR)

Amaç: Çözüm uzayının potansiyel olarak daha iyi keşfedilmesi ve uzun süren deneyler için kullanılabilirliğin artırılması amacıyla v1 üzerine özellikler ekleyerek iyileştirmek.
v1 Üzerindeki Ana İyileştirmeler:
Çaprazlama (Crossover): Mimari olarak uyumlu ebeveyn modeller arasında temel ağırlık çaprazlaması (ortalama/karıştırma) eklendi. Bu, farklı soylardan gelen potansiyel olarak iyi özellikleri birleştirmeyi hedefler.
Kontrol Noktası (Checkpointing): Evrim durumunu (popülasyon modelleri, rastgele durumlar) düzenli aralıklarla (--checkpoint_interval) kaydetme işlevselliği eklendi. --resume_from kullanarak yarıda kesilen çalıştırmalara devam etmeyi sağlar, bu özellikle CPU'lardaki uzun deneyler için kritiktir.
İyileştirilmiş Yapı: Kod yapısı, daha iyi sürdürülebilirlik ve yeni özelliklerin entegrasyonu için hafifçe iyileştirildi.
Beklenen Sonuç: Doğrulama için deneysel çalıştırmalar gerektirse de, v2'nin daha uzun evrim süreçleri için daha sağlam olması beklenmektedir. Çaprazlamanın eklenmesi, bazı durumlarda yerel optimalardan kaçmaya veya daha hızlı yakınsamaya yardımcı olabilir. Kontrol noktası özelliği, kapsamlı deneyler yapmanın pratikliğini önemli ölçüde artırır. Performans metrikleri (MSE, Tau) v2 çalıştırıldıktan sonra değerlendirilmelidir.
How to Run / Nasıl Çalıştırılır
(EN)

Clone the repository.
Install dependencies: pip install tensorflow numpy matplotlib scipy
To run v1:
Bash

python evonet_optimizer.py --generations 100 --pop_size 80 --seed 123
To run v2 (new run):
Bash

python evonet_optimizer_v3.py --generations 150 --pop_size 100 --crossover_rate 0.7 --checkpoint_interval 10 --seed 456
To resume v2 from a checkpoint:
Bash

# Assuming a previous run saved checkpoints in './evonet_runs_v3/evorun_...'
python evonet_optimizer_v3.py --resume_from ./evonet_runs_v3/evorun_... --generations 150 # Ensure total generations is correct
Results will be saved in timestamped subdirectories within evonet_runs... folders.
(TR)

Repo'yu klonlayın.
Bağımlılıkları yükleyin: pip install tensorflow numpy matplotlib scipy
v1'i çalıştırmak için:
Bash

python evonet_optimizer.py --generations 100 --pop_size 80 --seed 123
v2'yi çalıştırmak için (yeni çalıştırma):
Bash

python evonet_optimizer_v3.py --generations 150 --pop_size 100 --crossover_rate 0.7 --checkpoint_interval 10 --seed 456
v2'yi bir kontrol noktasından devam ettirmek için:
Bash

# Önceki bir çalıştırmanın checkpoint'leri './evonet_runs_v3/evorun_...' içine kaydettiğini varsayarsak
python evonet_optimizer_v3.py --resume_from ./evonet_runs_v3/evorun_... --generations 150 # Toplam nesil sayısının doğru olduğundan emin olun
Sonuçlar, evonet_runs... klasörleri içindeki zaman damgalı alt klasörlere kaydedilecektir.
Future Directions / Gelecek Yönelimler
(EN)

Implement more sophisticated neuroevolution algorithms like NEAT.
Evolve more complex network types (RNNs, Transformers).
Explore adaptive evolutionary parameters.
Apply the framework to more complex sequence processing tasks where NNs might offer greater advantages over classical methods.
Improve parallelization for faster execution.
(TR)

NEAT gibi daha sofistike neuroevolution algoritmalarını uygulamak.
Daha karmaşık ağ türlerini (RNN'ler, Transformer'lar) evrimleştirmek.
Uyarlanabilir evrimsel parametreleri keşfetmek.
Çerçeveyi, sinir ağlarının klasik yöntemlere göre daha fazla avantaj sunabileceği daha karmaşık dizi işleme görevlerine uygulamak.
Daha hızlı çalıştırma için paralelleştirmeyi iyileştirmek.
Disclaimer / Sorumluluk Reddi
(EN) This project is primarily for educational and experimental purposes. The neuroevolution approach used here is not expected to outperform highly optimized classical sorting algorithms (like Timsort) for general-purpose sorting in terms of speed or absolute numerical accuracy. Its value lies in exploring automatic architecture discovery via evolution. Performance depends heavily on configuration, random seed, and computational resources (CPU vs GPU).

(TR) Bu proje öncelikli olarak eğitim ve deneysel amaçlıdır. Burada kullanılan neuroevolution yaklaşımının, hız veya mutlak sayısal doğruluk açısından genel amaçlı sıralama için yüksek düzeyde optimize edilmiş klasik sıralama algoritmalarından (Timsort gibi) daha iyi performans göstermesi beklenmez. Değeri, evrim yoluyla otomatik mimari keşfini araştırmakta yatmaktadır. Performans, yapılandırmaya, rastgele tohuma ve hesaplama kaynaklarına (CPU vs GPU) büyük ölçüde bağlıdır.
