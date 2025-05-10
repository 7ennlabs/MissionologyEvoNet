# ==============================================================================
# EvoNet Optimizer - v4 - PyTorch Tabanlı Geliştirilmiş Sürüm
# Açıklama: TensorFlow'dan PyTorch'a geçiş yapılmış, modern PyTorch
#           pratikleri kullanılmış, esneklik artırılmış, kod kalitesi
#           iyileştirilmiş ve PyTorch ekosistemine uygun hale getirilmiştir.
#           Çaprazlama, Kontrol Noktası, Adaptif Mutasyon (kavramsal) ve
#           Gelişmiş Fitness (kavramsal) özellikleri korunmuştur.
# ==============================================================================

import os
import subprocess
import sys
import argparse
import random
import logging
from datetime import datetime
import json
import copy # Model klonlama ve durum dikteleri için
import time
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import kendalltau # Hala numpy/scipy kullanıyoruz

# --- Sabitler ve Varsayılan Değerler ---
DEFAULT_SEQ_LENGTH = 10
DEFAULT_POP_SIZE = 50
DEFAULT_GENERATIONS = 50
DEFAULT_CROSSOVER_RATE = 0.6
DEFAULT_MUTATION_RATE = 0.4 # Eğer çaprazlama olmazsa mutasyon olasılığı
DEFAULT_WEIGHT_MUT_RATE = 0.8 # Ağırlık mutasyonu olasılığı (mutasyon içinde)
# Aktivasyon mutasyonu PyTorch'ta daha farklı ele alınmalı, şimdilik odak ağırlıkta.
DEFAULT_MUTATION_STRENGTH = 0.1
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_ELITISM_COUNT = 2
DEFAULT_EPOCHS_FINAL_TRAIN = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "evonet_runs_v4_pytorch")
DEFAULT_CHECKPOINT_INTERVAL = 10 # Nesil başına checkpoint (0 = kapalı)
DEFAULT_DEVICE = "auto" # "auto", "cpu", "cuda"

# --- Loglama Ayarları ---
# (setup_logging fonksiyonu öncekiyle aynı, tekrar eklemiyorum)
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    log_filename = os.path.join(log_dir, 'evolution_run_pytorch.log')
    # Mevcut handler'ları temizle (özellikle tekrar çalıştırmalarda önemli)
    for handler in logging.root.handlers[:]:
        handler.close() # Önce kapat
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-8s [%(filename)s:%(lineno)d] - %(message)s', # Daha detaylı format
        handlers=[
            logging.FileHandler(log_filename, mode='a'), # append modu
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("="*50)
    logging.info("PyTorch EvoNet v4 Logging Başlatıldı.")
    logging.info("="*50)

# --- Cihaz (GPU/CPU) Ayarları ---
def setup_device(requested_device: str) -> torch.device:
    """ Kullanılabilir cihaza göre PyTorch cihazını ayarlar. """
    if requested_device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
            logging.info(f"CUDA (GPU) kullanılabilir: {torch.cuda.get_device_name(0)}")
        else:
            device_name = "cpu"
            logging.info("CUDA (GPU) bulunamadı. CPU kullanılacak.")
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            device_name = "cuda"
            logging.info(f"CUDA (GPU) manuel olarak seçildi: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("CUDA (GPU) istendi ancak bulunamadı! CPU kullanılacak.")
            device_name = "cpu"
    else: # cpu veya geçersiz değer
        device_name = "cpu"
        logging.info("CPU manuel olarak seçildi veya geçersiz cihaz belirtildi.")

    return torch.device(device_name)

# --- Veri Üretimi ---
# (generate_data fonksiyonu öncekiyle aynı, NumPy tabanlı)
def generate_data(num_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    logging.info(f"Generating {num_samples} samples with sequence length {seq_length}...")
    try:
        # Veriyi float32 olarak üretmek PyTorch için genellikle daha iyidir
        X = np.random.rand(num_samples, seq_length).astype(np.float32) * 100
        y = np.sort(X, axis=1).astype(np.float32)
        logging.info("Data generation successful.")
        return X, y
    except Exception as e:
        logging.error(f"Error during data generation: {e}", exc_info=True)
        raise

# --- PyTorch Sinir Ağı Modeli ---
class NeuralNetwork(nn.Module):
    """ Dinamik olarak yapılandırılabilen basit bir PyTorch MLP modeli. """
    def __init__(self, input_size: int, output_size: int, hidden_dims: List[int], activations: List[str]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims
        self.activations_str = activations # Mimarinin string listesi (checkpoint için)

        layers = []
        last_dim = input_size
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(last_dim, h_dim))
            act_func_str = activations[i].lower()
            if act_func_str == 'relu':
                layers.append(nn.ReLU())
            elif act_func_str == 'tanh':
                layers.append(nn.Tanh())
            elif act_func_str == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                logging.warning(f"Bilinmeyen aktivasyon '{activations[i]}', ReLU kullanılıyor.")
                layers.append(nn.ReLU()) # Varsayılan
            last_dim = h_dim

        # Çıkış katmanı (genellikle lineer aktivasyon)
        layers.append(nn.Linear(last_dim, output_size))

        self.network = nn.Sequential(*layers)
        self.architecture_id = self._generate_architecture_id() # Mimarinin özeti
        # Modelin adını (ID'sini) oluşturma (opsiyonel, loglama için kullanışlı)
        self.model_name = f"model_{self.architecture_id}_rnd{random.randint(10000, 99999)}"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_architecture(self) -> Dict[str, Any]:
        """ Model mimarisini döndürür (checkpointing için). """
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_dims": self.hidden_dims,
            "activations": self.activations_str
        }

    def _generate_architecture_id(self) -> str:
        """ Mimariden kısa bir kimlik üretir. """
        h_dims_str = '_'.join(map(str, self.hidden_dims))
        acts_str = ''.join([a[0].upper() for a in self.activations_str]) # R_T_S
        return f"I{self.input_size}_H{h_dims_str}_A{acts_str}_O{self.output_size}"

    # Eşitlik kontrolü mimari bazında yapılabilir
    def __eq__(self, other):
        if not isinstance(other, NeuralNetwork):
            return NotImplemented
        return self.get_architecture() == other.get_architecture()

    def __hash__(self):
         # Mimariyi temsil eden bir tuple oluştur ve hash'ini al
         arch_tuple = (
             self.input_size,
             self.output_size,
             tuple(self.hidden_dims),
             tuple(self.activations_str)
         )
         return hash(arch_tuple)


# --- Neuroevolution Çekirdeği (PyTorch) ---

def create_individual_pytorch(input_size: int, output_size: int) -> NeuralNetwork:
    """ Rastgele mimariye sahip bir PyTorch NeuralNetwork modeli oluşturur. """
    try:
        num_hidden_layers = random.randint(1, 4)
        hidden_dims = [random.randint(16, 128) for _ in range(num_hidden_layers)] # Biraz daha geniş aralık
        activations = [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(num_hidden_layers)]

        model = NeuralNetwork(input_size, output_size, hidden_dims, activations)
        # PyTorch'ta model oluşturulduktan sonra compile gerekmez.
        # Ağırlıklar zaten rastgele başlatılır.
        logging.debug(f"Created individual: {model.model_name}")
        return model
    except Exception as e:
        logging.error(f"Error creating PyTorch individual model: {e}", exc_info=True)
        raise

# PyTorch için model kopyalama işlevi
def clone_pytorch_model(model: NeuralNetwork, device: torch.device) -> NeuralNetwork:
    """ Bir PyTorch modelini (mimari ve ağırlıklar) klonlar. """
    try:
        # 1. Aynı mimariyle yeni bir model oluştur
        arch = model.get_architecture()
        cloned_model = NeuralNetwork(**arch)
        # 2. Orijinal modelin state_dict'ini kopyala
        cloned_model.load_state_dict(copy.deepcopy(model.state_dict()))
        # 3. Yeni modeli doğru cihaza taşı
        cloned_model.to(device)
        cloned_model.model_name = f"cloned_{model.model_name}_{random.randint(1000,9999)}"
        logging.debug(f"Cloned model {model.model_name} to {cloned_model.model_name}")
        return cloned_model
    except Exception as e:
        logging.error(f"Error cloning PyTorch model {model.model_name}: {e}", exc_info=True)
        raise

def calculate_fitness_pytorch(
    individual: NeuralNetwork,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    fitness_params: Optional[Dict] = None
) -> float:
    """ Bir bireyin fitness değerini PyTorch kullanarak hesaplar. """
    # --- KAVRAMSAL: Gelişmiş Fitness Fonksiyonu (PyTorch ile uyumlu) ---
    # fitness_params = fitness_params or {}
    # w_mse = fitness_params.get('w_mse', 1.0)
    # w_tau = fitness_params.get('w_tau', 0.1)  # Kendall Tau ağırlığı
    # w_comp = fitness_params.get('w_comp', 0.0001) # Karmaşıklık cezası ağırlığı
    # --------------------------------------------

    individual.eval() # Modeli değerlendirme moduna al (dropout vs. etkisizleşir)
    individual.to(device) # Modeli doğru cihaza taşı
    X, y = X.to(device), y.to(device) # Veriyi doğru cihaza taşı

    try:
        with torch.no_grad(): # Gradyan hesaplamasını kapat (inferans için)
            y_pred = individual(X)
            # Temel Fitness: MSE (Mean Squared Error)
            # loss_fn = nn.MSELoss()
            # mse_val = loss_fn(y_pred, y).item()
            # Alternatif manuel hesaplama:
            mse_val = torch.mean((y_pred - y)**2).item()

        # MSE sonsuz veya NaN ise minimum fitness ata
        if not np.isfinite(mse_val):
            logging.warning(f"Non-finite MSE ({mse_val}) for model {individual.model_name}. Assigning minimal fitness.")
            return -1e9 # Çok düşük bir değer ata

        # Temel Fitness (MSE'nin tersi, daha yüksek daha iyi)
        fitness_score = 1.0 / (mse_val + 1e-9) # Sıfıra bölme hatasını önle

        # --- KAVRAMSAL: Gelişmiş Fitness Hesabı ---
        # if w_tau > 0 or w_comp > 0:
        #     # Kendall Tau hesapla (NumPy'a çevirerek, maliyetli olabilir)
        #     y_np = y.cpu().numpy()
        #     y_pred_np = y_pred.cpu().numpy()
        #     tau_val = calculate_avg_kendall_tau(y_np, y_pred_np, sample_size=100) # Örnek fonksiyon
        #
        #     # Karmaşıklık hesapla (parametre sayısı)
        #     complexity = sum(p.numel() for p in individual.parameters() if p.requires_grad)
        #
        #     # Birleştirilmiş fitness (Örnek formül)
        #     # MSE'yi minimize etmek istediğimiz için 1/MSE kullanıyoruz.
        #     # Tau'yu maksimize etmek istiyoruz.
        #     # Karmaşıklığı minimize etmek istiyoruz.
        #     fitness_score = (w_mse * fitness_score) + (w_tau * tau_val) - (w_comp * complexity)
        # --------------------------------------------

        # Sonuçta yine de çok düşük veya sonsuz fitness kontrolü
        if not np.isfinite(fitness_score) or fitness_score < -1e8:
             logging.warning(f"Non-finite or very low final fitness ({fitness_score:.4g}) for model {individual.model_name}. Assigning minimal fitness.")
             return -1e9

        return float(fitness_score)

    except Exception as e:
        logging.error(f"Error during fitness calculation for model {individual.model_name}: {e}", exc_info=True)
        return -1e9 # Hata durumunda çok düşük fitness


def mutate_individual_pytorch(
    individual: NeuralNetwork,
    weight_mut_rate: float, # Bu parametre aslında ağırlıkların *ne kadarının* mutasyona uğrayacağını belirleyebilir
    mutation_strength: float,
    device: torch.device
) -> NeuralNetwork:
    """ Bir PyTorch bireyine ağırlık bozulması mutasyonu uygular. """
    try:
        # Önemli: Orijinal modeli değiştirmemek için klonla
        mutated_model = clone_pytorch_model(individual, device)
        mutated_model.model_name = f"mutated_{individual.model_name}_{random.randint(1000,9999)}"

        mutated = False
        # Modelin state_dict'i üzerinde değişiklik yap
        state_dict = mutated_model.state_dict()
        new_state_dict = copy.deepcopy(state_dict) # Derin kopya al

        for name, param in new_state_dict.items():
            # Sadece eğitilebilir ağırlık/bias tensörlerini değiştir
            if param.requires_grad and random.random() < weight_mut_rate : # Her parametre için mutasyon olasılığı
                 mutated = True
                 # Gaussian gürültü ekle
                 noise = torch.randn_like(param) * mutation_strength
                 new_state_dict[name] = param + noise.to(param.device) # Gürültüyü doğru cihaza taşı

        if mutated:
            mutated_model.load_state_dict(new_state_dict)
            logging.debug(f"Mutated model {individual.model_name} -> {mutated_model.model_name}")
            return mutated_model
        else:
            # Mutasyon uygulanmadıysa, klonlanmış modeli (isim değiştirilmiş) döndür veya orijinali?
            # Mantıksal olarak mutasyon fonksiyonu çağrıldıysa bir değişiklik beklenir.
            # Eğer hiç parametre mutasyona uğramadıysa bile farklı bir obje döndürmek tutarlı olabilir.
            logging.debug(f"Mutation applied to {individual.model_name}, but no weights changed based on rate.")
            return mutated_model # Klonlanmış, potansiyel olarak ismi değişmiş modeli döndür

    except Exception as e:
        logging.error(f"Error during PyTorch mutation of model {individual.model_name}: {e}", exc_info=True)
        # Hata durumunda orijinal bireyi döndürmek güvenli bir seçenek olabilir
        # return individual
        # Ancak evrimsel süreçte sorun yaratabilir, bu yüzden klonlanmışı döndürmek daha iyi
        return clone_pytorch_model(individual, device) # Hata durumunda temiz klon döndür


def check_architecture_compatibility_pytorch(model1: NeuralNetwork, model2: NeuralNetwork) -> bool:
    """ İki PyTorch modelinin basit çaprazlama için uyumlu olup olmadığını kontrol eder. """
    # Mimari bilgilerini karşılaştır
    return model1.get_architecture() == model2.get_architecture()


def crossover_individuals_pytorch(
    parent1: NeuralNetwork,
    parent2: NeuralNetwork,
    device: torch.device
) -> Tuple[Optional[NeuralNetwork], Optional[NeuralNetwork]]:
    """ İki PyTorch ebeveynden basit ağırlık ortalaması/karıştırması ile çocuklar oluşturur. """

    # 1. Mimari uyumluluğunu kontrol et
    if not check_architecture_compatibility_pytorch(parent1, parent2):
        logging.debug(f"Skipping crossover between {parent1.model_name} and {parent2.model_name} due to incompatible architectures.")
        return None, None

    try:
        # 2. Çocuklar için yeni model örnekleri oluştur (aynı mimariyle)
        arch = parent1.get_architecture() # İkisi de aynı mimariye sahip
        child1 = NeuralNetwork(**arch).to(device)
        child2 = NeuralNetwork(**arch).to(device)
        child1.model_name = f"xover_{parent1.architecture_id}_c1_{random.randint(1000,9999)}"
        child2.model_name = f"xover_{parent1.architecture_id}_c2_{random.randint(1000,9999)}"


        # 3. Ebeveynlerin state_dict'lerini al
        p1_state = parent1.state_dict()
        p2_state = parent2.state_dict()

        # 4. Çocukların state_dict'lerini oluştur
        c1_state = child1.state_dict() # Başlangıç (rastgele) state'i al
        c2_state = child2.state_dict()

        for name in p1_state: # Parametre isimleri üzerinden döngü
            param1 = p1_state[name]
            param2 = p2_state[name]

            # Basit ortalama çaprazlama (daha fazla yöntem eklenebilir)
            # c1_state[name] = (param1 + param2) / 2.0
            # c2_state[name] = (param1 + param2) / 2.0 # Ortalama için ikisi de aynı

            # Tek nokta veya uniform crossover (ağırlık matrisi üzerinde)
            mask = torch.rand_like(param1) < 0.5
            c1_state[name] = torch.where(mask, param1, param2)
            c2_state[name] = torch.where(mask, param2, param1) # Ters maske ile

        # 5. Yeni state_dict'leri çocuklara yükle
        child1.load_state_dict(c1_state)
        child2.load_state_dict(c2_state)

        logging.debug(f"Crossover performed between {parent1.model_name} and {parent2.model_name}")
        return child1, child2

    except Exception as e:
        logging.error(f"Error during PyTorch crossover between {parent1.model_name} and {parent2.model_name}: {e}", exc_info=True)
        return None, None

# (tournament_selection fonksiyonu öncekiyle aynı mantıkta çalışır, sadece model yerine
#  NeuralNetwork objesini döndürür)
def tournament_selection(
    population: List[NeuralNetwork],
    fitness_scores: List[float],
    k: int
) -> NeuralNetwork:
    """ Turnuva seçimi ile popülasyondan bir birey seçer. """
    if not population:
        raise ValueError("Population cannot be empty for tournament selection.")
    if len(population) < k:
        logging.warning(f"Tournament size ({k}) is larger than population size ({len(population)}). Using population size.")
        k = len(population)
    if k <= 0:
        logging.warning(f"Tournament size ({k}) must be positive. Using 1.")
        k = 1

    try:
        # Popülasyondan k bireyi rastgele seç (indeksleriyle)
        tournament_indices = random.sample(range(len(population)), k)
        # Seçilenlerin fitness skorlarını ve kendilerini al
        tournament_contenders = [(fitness_scores[i], population[i]) for i in tournament_indices]
        # Fitness'a göre en iyiyi seç
        winner = max(tournament_contenders, key=lambda item: item[0])[1] # item[0] fitness, item[1] model
        return winner
    except Exception as e:
        logging.error(f"Error during tournament selection: {e}", exc_info=True)
        # Hata durumunda rastgele bir birey döndür
        return random.choice(population)


# --- Checkpointing (PyTorch) ---
def save_checkpoint_pytorch(output_dir: str, generation: int, population: List[NeuralNetwork], rnd_state: Any, np_rnd_state: Any, torch_rnd_state: Any):
    """ Evrim durumunu (PyTorch modelleri ve rastgele durumlar) kaydeder. """
    checkpoint_dir = os.path.join(output_dir, "checkpoints_pytorch")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"evo_gen_{generation}.pt") # .pt uzantısı PyTorch için yaygın
    logging.info(f"Saving checkpoint for generation {generation} to {checkpoint_file}...")

    population_state = []
    for model in population:
        try:
            # Her model için mimariyi ve state_dict'i kaydet
            population_state.append({
                "name": model.model_name,
                "architecture": model.get_architecture(),
                "state_dict": model.state_dict()
            })
        except Exception as e:
            logging.error(f"Could not serialize model {model.model_name} for checkpoint: {e}")
            # Başarısız olursa bu modeli atla

    state = {
        "generation": generation,
        "population_state": population_state, # Sadece başarılı olanları içerir
        "random_state": rnd_state,
        "numpy_random_state": np_rnd_state,
        "torch_random_state": torch_rnd_state, # PyTorch RNG durumu
        "timestamp": datetime.now().isoformat()
    }

    try:
        torch.save(state, checkpoint_file)
        logging.info(f"Checkpoint saved successfully for generation {generation}.")
    except Exception as e:
        logging.error(f"Failed to save checkpoint using torch.save for generation {generation}: {e}", exc_info=True)


def load_checkpoint_pytorch(checkpoint_path: str, device: torch.device) -> Optional[Dict]:
    """ Kaydedilmiş PyTorch evrim durumunu yükler. """
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        return None
    logging.info(f"Loading checkpoint from {checkpoint_path}...")

    try:
        # Checkpoint'i CPU'ya yüklemek genellikle daha güvenlidir, sonra cihaza taşınır
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        population = []
        for model_state in checkpoint["population_state"]:
            try:
                # 1. Mimariden modeli yeniden oluştur
                arch = model_state["architecture"]
                model = NeuralNetwork(**arch)
                # 2. Kaydedilmiş state_dict'i yükle
                model.load_state_dict(model_state["state_dict"])
                # 3. Modeli istenen cihaza taşı
                model.to(device)
                # 4. Model adını geri yükle (opsiyonel)
                model.model_name = model_state.get("name", f"loaded_model_{random.randint(1000,9999)}")
                model.eval() # Değerlendirme modunda başlat
                population.append(model)
            except Exception as e:
                logging.error(f"Failed to load model state from checkpoint for model {model_state.get('name', 'UNKNOWN')}: {e}", exc_info=True)

        if not population:
            logging.error("Failed to load any model from the checkpoint population state.")
            return None # Hiç model yüklenemediyse checkpoint geçersiz

        # Yüklenen popülasyonu state'e ekle
        checkpoint["population"] = population

        logging.info(f"Checkpoint loaded successfully. Resuming from generation {checkpoint['generation'] + 1}.")
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
        return None

def find_latest_checkpoint_pytorch(output_dir: str) -> Optional[str]:
    """ Verilen klasördeki en son PyTorch checkpoint dosyasını (.pt) bulur. """
    checkpoint_dir = os.path.join(output_dir, "checkpoints_pytorch")
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("evo_gen_") and f.endswith(".pt")]
    if not checkpoints:
        return None

    latest_gen = -1
    latest_file = None
    for cp in checkpoints:
        try:
            gen_num = int(cp.split('_')[2].split('.')[0])
            if gen_num > latest_gen:
                latest_gen = gen_num
                latest_file = os.path.join(checkpoint_dir, cp)
        except (IndexError, ValueError):
            logging.warning(f"Could not parse generation number from checkpoint file: {cp}")
            continue
    return latest_file


# --- Ana Evrim Döngüsü (PyTorch) ---
def evolve_population_pytorch(
    population: List[NeuralNetwork],
    X: np.ndarray, y: np.ndarray, # Veri hala NumPy olarak geliyor
    start_generation: int, total_generations: int,
    crossover_rate: float, mutation_rate: float, weight_mut_rate: float, mut_strength: float,
    tournament_size: int, elitism_count: int, batch_size: int, # batch_size fitness'ta kullanılmıyor şu an
    output_dir: str, checkpoint_interval: int, device: torch.device
) -> Tuple[Optional[NeuralNetwork], List[float], List[float]]:
    """ PyTorch tabanlı evrimsel süreci çalıştırır. """

    best_fitness_history = []
    avg_fitness_history = []
    best_model_overall: Optional[NeuralNetwork] = None
    best_fitness_overall = -np.inf

    # Veriyi PyTorch tensörlerine dönüştür ve cihaza gönder (bir kere)
    # Büyük veri setleri için DataLoader düşünülebilir, ancak burada basit tutuyoruz
    try:
        X_torch = torch.from_numpy(X).float().to(device)
        y_torch = torch.from_numpy(y).float().to(device)
    except Exception as e:
        logging.critical(f"Failed to convert data to PyTorch tensors or move to device: {e}", exc_info=True)
        raise

    # --- KAVRAMSAL: Uyarlanabilir Mutasyon Oranı (Adaptif Parametreler) ---
    # current_mutation_strength = mut_strength
    # stagnation_counter = 0
    # stagnation_limit = 10 # Örneğin, 10 nesil iyileşme olmazsa...
    # min_mut_strength = 0.01
    # max_mut_strength = 0.5
    # --------------------------------------------

    pop_size = len(population)

    for gen in range(start_generation, total_generations):
        generation_start_time = time.time()

        # 1. Fitness Değerlendirme
        try:
            # Paralelleştirme potansiyeli (eğer fitness hesaplama çok uzun sürüyorsa)
            # Örnek: concurrent.futures kullanarak
            fitness_scores = [calculate_fitness_pytorch(ind, X_torch, y_torch, device) for ind in population]
        except Exception as e:
            logging.critical(f"Error calculating fitness for population in Generation {gen+1}: {e}", exc_info=True)
            # Hata durumunda en iyi modeli döndürmeye çalış
            if best_model_overall:
                return best_model_overall, best_fitness_history, avg_fitness_history
            else:
                raise # Eğer hiç en iyi model yoksa, hata ver

        # 2. İstatistikler ve En İyiyi Takip
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        # NaN veya Inf değerlerini filtreleyerek ortalama hesapla
        finite_scores = [s for s in fitness_scores if np.isfinite(s)]
        avg_fitness = np.mean(finite_scores) if finite_scores else -np.inf

        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)

        new_best_found = False
        if current_best_fitness > best_fitness_overall and np.isfinite(current_best_fitness):
            best_fitness_overall = current_best_fitness
            new_best_found = True
            try:
                # En iyi modeli klonla (orijinal popülasyondaki değişmesin)
                best_model_overall = clone_pytorch_model(population[current_best_idx], device)
                logging.info(f"Generation {gen+1}: *** New overall best fitness found: {best_fitness_overall:.6f} (Model: {best_model_overall.model_name}) ***")
            except Exception as e:
                logging.error(f"Could not clone new best model {population[current_best_idx].model_name}: {e}", exc_info=True)
                # Klonlama başarısız olursa, en azından fitness'ı takip et
                best_model_overall = None # Klonlanamadığı için referansı tutma
        # else: # En iyi bulunamadıysa veya aynıysa
             # --- KAVRAMSAL: Adaptif Mutasyon Güncelleme ---
             # stagnation_counter += 1
             # logging.debug(f"Stagnation counter: {stagnation_counter}")
             # if stagnation_counter >= stagnation_limit:
             #      current_mutation_strength = min(max_mut_strength, current_mutation_strength * 1.2) # Mutasyon gücünü artır
             #      logging.info(f"Stagnation detected. Increasing mutation strength to {current_mutation_strength:.4f}")
             #      stagnation_counter = 0 # Sayacı sıfırla

        # if new_best_found:
        #      stagnation_counter = 0
        #      current_mutation_strength = max(min_mut_strength, current_mutation_strength * 0.95) # İyileşme varsa azalt
        #      logging.debug(f"Improvement found. Decreasing mutation strength to {current_mutation_strength:.4f}")

        generation_time = time.time() - generation_start_time
        logging.info(f"Generation {gen+1}/{total_generations} | Best Fitness: {current_best_fitness:.6f} | Avg Fitness: {avg_fitness:.6f} | Time: {generation_time:.2f}s")

        # 3. Yeni Popülasyon Oluşturma
        new_population = []

        # 3a. Elitizm
        if elitism_count > 0 and len(population) >= elitism_count:
            try:
                # Fitness skorlarına göre sırala ve en iyileri al (indeksleri)
                elite_indices = np.argsort(fitness_scores)[-elitism_count:]
                for idx in elite_indices:
                    # Elitleri klonlayarak yeni popülasyona ekle
                    elite_clone = clone_pytorch_model(population[idx], device)
                    elite_clone.model_name = f"elite_{population[idx].model_name}" # İsimlendirme
                    new_population.append(elite_clone)
                logging.debug(f"Added {len(new_population)} elites to the next generation.")
            except Exception as e:
                logging.error(f"Error during elitism: {e}", exc_info=True)

        # 3b. Seçilim, Çaprazlama ve Mutasyon ile kalanları doldur
        num_to_generate = pop_size - len(new_population)
        generated_count = 0
        reproduction_attempts = 0 # Sonsuz döngüyü önlemek için
        max_reproduction_attempts = num_to_generate * 5 # Cömert bir sınır

        while generated_count < num_to_generate and reproduction_attempts < max_reproduction_attempts:
            reproduction_attempts += 1
            try:
                # İki ebeveyn seç
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)

                child1, child2 = None, None

                # Çaprazlama uygula (belirli bir olasılıkla ve farklı ebeveynlerse)
                if random.random() < crossover_rate and parent1 is not parent2:
                     # logging.debug(f"Attempting crossover between {parent1.model_name} and {parent2.model_name}")
                     child1, child2 = crossover_individuals_pytorch(parent1, parent2, device)

                # Eğer çaprazlama yapılmadıysa/başarısız olduysa veya tek çocuk üretildiyse
                if child1 is None:
                    # Mutasyon uygula (belirli bir olasılıkla)
                    if random.random() < mutation_rate:
                        parent_to_mutate = parent1 # Veya parent2, veya rastgele biri
                        child1 = mutate_individual_pytorch(parent_to_mutate, weight_mut_rate, mut_strength, device) # Adaptif: current_mutation_strength
                    else:
                        # Ne çaprazlama ne mutasyon -> ebeveyni klonla
                        child1 = clone_pytorch_model(parent1, device)
                        child1.model_name = f"direct_clone_{parent1.model_name}_{random.randint(1000,9999)}"

                # Çocukları yeni popülasyona ekle (eğer üretildilerse)
                if child1:
                    new_population.append(child1)
                    generated_count += 1
                    if generated_count >= num_to_generate: break

                if child2: # Eğer çaprazlama iki çocuk ürettiyse
                    # İkinci çocuğa da mutasyon uygulama seçeneği eklenebilir
                    # if random.random() < post_crossover_mutation_rate: child2 = mutate(...)
                    new_population.append(child2)
                    generated_count += 1
                    if generated_count >= num_to_generate: break

            except Exception as e:
                logging.error(f"Error during selection/reproduction cycle (attempt {reproduction_attempts}): {e}", exc_info=True)
                # Hata durumunda döngüye devam etmeye çalış, ancak sınırı aşarsa durur.
                # Güvenlik önlemi olarak rastgele birey eklenebilir ama hatayı maskeleyebilir.

        # Eğer döngü sınırı aşıldıysa popülasyonu tamamla
        if generated_count < num_to_generate:
            logging.warning(f"Reproduction cycle finished early or hit attempt limit. Adding {num_to_generate - generated_count} random individuals.")
            input_size = population[0].input_size # İlk bireyden al
            output_size = population[0].output_size
            for _ in range(num_to_generate - generated_count):
                try:
                    random_ind = create_individual_pytorch(input_size, output_size).to(device)
                    new_population.append(random_ind)
                except Exception as e:
                     logging.error(f"Failed to create random individual to fill population: {e}")
                     # Bu durumda popülasyon eksik kalabilir

        population = new_population[:pop_size] # Popülasyon boyutunu garantile

        # 4. Checkpoint Alma
        if checkpoint_interval > 0 and (gen + 1) % checkpoint_interval == 0:
            try:
                rnd_state = random.getstate()
                np_rnd_state = np.random.get_state()
                torch_rnd_state = torch.get_rng_state() # PyTorch RNG durumu
                # Cihaz RNG durumları da kaydedilebilir: torch.cuda.get_rng_state_all()
                save_checkpoint_pytorch(output_dir, gen + 1, population, rnd_state, np_rnd_state, torch_rnd_state)
            except Exception as e:
                logging.error(f"Failed to execute checkpoint saving for generation {gen+1}: {e}", exc_info=True)

        # Döngü sonu temizliği (GPU belleği için önemli olabilir)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Evrim Döngüsü Sonu
    if best_model_overall is None:
        logging.warning("Evolution finished, but no single best model was tracked (possibly due to errors or all fitness being non-finite).")
        # Son popülasyondan en iyiyi bulmaya çalış
        if population:
             final_fitness_scores = [calculate_fitness_pytorch(ind, X_torch, y_torch, device) for ind in population]
             valid_scores = [(s, i) for i, s in enumerate(final_fitness_scores) if np.isfinite(s)]
             if valid_scores:
                 best_idx_final = max(valid_scores, key=lambda item: item[0])[1]
                 best_model_overall = clone_pytorch_model(population[best_idx_final], device) # Klonla
                 best_fitness_overall = final_fitness_scores[best_idx_final]
                 logging.info(f"Selected best model from final population: {best_model_overall.model_name} with fitness {best_fitness_overall:.6f}")
             else:
                 logging.error("Evolution finished. No valid finite fitness scores in the final population.")
                 return None, best_fitness_history, avg_fitness_history
        else:
             logging.error("Evolution finished with an empty population!")
             return None, best_fitness_history, avg_fitness_history
    else:
         logging.info(f"Evolution finished. Best fitness achieved: {best_fitness_overall:.6f} by model {best_model_overall.model_name}")

    return best_model_overall, best_fitness_history, avg_fitness_history

# --- Grafik Çizimi (Öncekiyle aynı, Matplotlib kullanıyor) ---
def plot_fitness_history(history_best: List[float], history_avg: List[float], output_dir: str, filename: str = "fitness_history_pytorch.png") -> None:
    if not history_best or not history_avg:
        logging.warning("Fitness history is empty, cannot plot.")
        return
    try:
        plt.figure(figsize=(12, 7))
        # NaN veya Inf değerlerini çizimde atlamak için filtrele
        gens = np.arange(1, len(history_best) + 1)
        valid_best_indices = [i for i, v in enumerate(history_best) if np.isfinite(v)]
        valid_avg_indices = [i for i, v in enumerate(history_avg) if np.isfinite(v)]

        if valid_best_indices:
             plt.plot(gens[valid_best_indices], np.array(history_best)[valid_best_indices], label="Best Fitness", marker='o', linestyle='-', linewidth=2)
        if valid_avg_indices:
             plt.plot(gens[valid_avg_indices], np.array(history_avg)[valid_avg_indices], label="Average Fitness", marker='x', linestyle='--', alpha=0.7)

        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.title("Evolutionary Fitness History (PyTorch)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path)
        plt.close() # Belleği boşalt
        logging.info(f"Fitness history plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Error plotting fitness history: {e}", exc_info=True)


# --- Değerlendirme (PyTorch) ---
def evaluate_model_pytorch(
    model: NeuralNetwork,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int, device: torch.device
) -> Dict[str, float]:
    """ En iyi modeli test verisi üzerinde PyTorch ile değerlendirir. """
    if model is None:
        logging.error("Cannot evaluate a None model.")
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}

    logging.info("Evaluating final model on test data using PyTorch...")
    model.eval() # Değerlendirme modu
    model.to(device)

    # NumPy verisini PyTorch DataLoader ile kullanmak
    try:
        test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        test_loader = DataLoader(test_dataset, batch_size=batch_size) # Shuffle=False önemli
    except Exception as e:
        logging.error(f"Failed to create PyTorch DataLoader for test data: {e}", exc_info=True)
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}

    all_preds = []
    all_targets = []
    total_mse = 0.0
    num_batches = 0

    try:
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                batch_mse = torch.mean((outputs - targets)**2)
                total_mse += batch_mse.item()
                num_batches += 1
                # Kendall Tau için tahminleri ve hedefleri topla (CPU'da)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_mse = total_mse / num_batches if num_batches > 0 else np.inf
        logging.info(f"Final Test MSE: {avg_mse:.6f}")

        # Kendall Tau hesaplaması
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)

        sample_size = min(500, all_targets_np.shape[0])
        taus = []
        if sample_size > 0:
            indices = np.random.choice(all_targets_np.shape[0], sample_size, replace=False)
            for i in indices:
                try:
                    tau, _ = kendalltau(all_targets_np[i], all_preds_np[i])
                    if not np.isnan(tau):
                        taus.append(tau)
                except ValueError: # Sabit tahmin durumu vb.
                    pass
        avg_kendall_tau = np.mean(taus) if taus else 0.0
        logging.info(f"Average Kendall's Tau (on {sample_size} samples): {avg_kendall_tau:.4f}")

        return {"test_mse": float(avg_mse), "avg_kendall_tau": float(avg_kendall_tau)}

    except Exception as e:
        logging.error(f"Error during final PyTorch model evaluation: {e}", exc_info=True)
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}


# --- Son Eğitim (PyTorch) ---
def train_final_model_pytorch(
    model: NeuralNetwork,
    X_train: np.ndarray, y_train: np.ndarray,
    epochs: int, batch_size: int, learning_rate: float,
    device: torch.device, output_dir: str
) -> Tuple[NeuralNetwork, Dict[str, Any]]:
    """ En iyi evrimleşmiş modeli PyTorch ile eğitir (Early Stopping ve LR Scheduling ile). """
    logging.info(f"--- Starting Final Training of Best Evolved Model ({model.model_name}) ---")
    model.to(device) # Modeli cihaza taşı

    # Veriyi DataLoader'a yükle
    try:
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        # Veriyi train/validation olarak ayır
        val_split = 0.2
        num_train = len(train_dataset)
        split_idx = int(np.floor(val_split * num_train))
        indices = list(range(num_train))
        np.random.shuffle(indices) # Karıştır
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices) # Veya SequentialSampler

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
        logging.info(f"Created DataLoaders. Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    except Exception as e:
        logging.error(f"Failed to create DataLoaders for final training: {e}", exc_info=True)
        return model, {"error": "DataLoader creation failed"}

    # Optimizatör ve Kayıp Fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Kayıp fonksiyonu

    # Learning Rate Scheduler (Platoda Azaltma)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=7, verbose=True, min_lr=1e-7)

    # Early Stopping Parametreleri
    early_stopping_patience = 15
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None # En iyi modelin state_dict'ini sakla

    training_history = {'train_loss': [], 'val_loss': [], 'lr': []}
    epochs_run = 0

    try:
        for epoch in range(epochs):
            epochs_run += 1
            model.train() # Eğitim modu
            running_train_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()       # Gradyanları sıfırla
                outputs = model(inputs)     # İleri besleme
                loss = criterion(outputs, targets) # Kaybı hesapla
                loss.backward()             # Geri yayılım
                optimizer.step()            # Ağırlıkları güncelle

                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            training_history['train_loss'].append(avg_train_loss)
            training_history['lr'].append(optimizer.param_groups[0]['lr']) # Mevcut LR'yi kaydet

            # ---- Validation ----
            model.eval() # Değerlendirme modu
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else np.inf
            training_history['val_loss'].append(avg_val_loss)

            logging.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Learning Rate Scheduling
            scheduler.step(avg_val_loss)

            # Early Stopping Kontrolü
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # En iyi modelin durumunu kaydet (derin kopya)
                best_model_state = copy.deepcopy(model.state_dict())
                logging.debug(f"New best validation loss: {best_val_loss:.6f}. Saving model state.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {early_stopping_patience} epochs.")
                break

        # Eğitim sonrası en iyi modeli yükle (eğer kaydedildiyse)
        if best_model_state:
            logging.info(f"Restoring model to best validation performance (Val Loss: {best_val_loss:.6f}).")
            model.load_state_dict(best_model_state)
        else:
             logging.warning("No best model state was saved during training (possibly validation loss never improved).")


        logging.info("Final training complete.")
        training_summary = {
            "epochs_run": epochs_run,
            "final_train_loss": avg_train_loss, # Son epoch'un kaybı
            "best_val_loss": best_val_loss, # Elde edilen en iyi val kaybı
            "final_lr": optimizer.param_groups[0]['lr']
        }
        # Eğitim grafiğini çizdir (opsiyonel)
        # plot_training_history(training_history, output_dir)

        return model, training_summary

    except Exception as e:
        logging.error(f"Error during final PyTorch model training: {e}", exc_info=True)
        return model, {"error": str(e)}


# --- Ana İş Akışı (PyTorch) ---
def run_pipeline_pytorch(args: argparse.Namespace):
    """ Checkpoint ve PyTorch tabanlı ana iş akışı. """

    # Cihazı Ayarla
    device = setup_device(args.device)

    # Çalıştırma adı ve çıktı klasörü
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evorun_pt_{timestamp}_gen{args.generations}_pop{args.pop_size}"
    output_dir = args.resume_from if args.resume_from else os.path.join(args.output_base_dir, run_name)
    resume_run = bool(args.resume_from)

    if resume_run:
        run_name = os.path.basename(output_dir)
        logging.info(f"Attempting to resume PyTorch run from: {output_dir}")
        # Devam edilen çalıştırmada çıktı klasörü zaten var olmalı
        if not os.path.isdir(output_dir):
             logging.error(f"Resume directory not found: {output_dir}. Exiting.")
             sys.exit(1)
    else:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"FATAL: Could not create output directory: {output_dir}. Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Loglamayı ayarla ('a' modu ile devam etmeye uygun)
    setup_logging(output_dir)
    logging.info(f"========== Starting/Resuming EvoNet v4 PyTorch Pipeline: {run_name} ==========")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Using device: {device}")

    # --- Checkpoint Yükleme ---
    start_generation = 0
    population = []
    initial_state_loaded = False
    loaded_history_best = [] # Yüklenecek geçmiş fitness verileri
    loaded_history_avg = []

    latest_checkpoint_path = find_latest_checkpoint_pytorch(output_dir) if resume_run else None

    if latest_checkpoint_path:
        loaded_state = load_checkpoint_pytorch(latest_checkpoint_path, device)
        if loaded_state:
            start_generation = loaded_state['generation']
            population = loaded_state['population'] # Yüklenen modeller zaten doğru cihazda olmalı
            # Rastgele durumları geri yükle
            try:
                random.setstate(loaded_state['random_state'])
                np.random.set_state(loaded_state['numpy_random_state'])
                torch.set_rng_state(loaded_state['torch_random_state'].cpu()) # CPU'ya yüklenen state'i kullan
                if device.type == 'cuda' and 'torch_cuda_random_state' in loaded_state:
                    # TODO: CUDA RNG state'i de kaydet/yükle (gerekirse)
                    # torch.cuda.set_rng_state_all(loaded_state['torch_cuda_random_state'])
                    pass
                logging.info(f"Random states restored from checkpoint (Generation {start_generation}).")
            except Exception as e:
                logging.warning(f"Could not fully restore random states from checkpoint: {e}")

            # TODO: Fitness geçmişini de checkpoint'e kaydet/yükle
            # loaded_history_best = loaded_state.get('best_fitness_history', [])
            # loaded_history_avg = loaded_state.get('avg_fitness_history', [])

            initial_state_loaded = True
            logging.info(f"Resuming from Generation {start_generation + 1} with {len(population)} individuals.")
        else:
            logging.error("Failed to load checkpoint. Starting from scratch.")
            resume_run = False
    elif resume_run:
        logging.warning(f"Resume requested but no valid PyTorch checkpoint (.pt) found in {output_dir}. Starting from scratch.")
        resume_run = False


    # --- Sıfırdan Başlama veya Devam Etme Ayarları ---
    # Argümanları logla ve kaydet (sadece sıfırdan başlarken veya config yoksa)
    config_path = os.path.join(output_dir, "config_pytorch.json")
    args_dict = vars(args)
    if not initial_state_loaded or not os.path.exists(config_path):
         logging.info("--- Configuration ---")
         for k, v in args_dict.items(): logging.info(f"  {k:<25}: {v}")
         logging.info("---------------------")
         try:
             # Argümanları JSON olarak kaydet
             args_to_save = args_dict.copy()
             # Cihaz objesini string'e çevir
             args_to_save['device'] = str(device)
             with open(config_path, 'w') as f: json.dump(args_to_save, f, indent=4, sort_keys=True)
             logging.info(f"Configuration saved to {config_path}")
         except Exception as e: logging.error(f"Failed to save configuration: {e}", exc_info=True)
    else: # Devam ediliyorsa ve config varsa, onu logla
         try:
              with open(config_path, 'r') as f: loaded_args_dict = json.load(f)
              logging.info("--- Loaded Configuration (from resumed run) ---")
              for k, v in loaded_args_dict.items(): logging.info(f"  {k:<25}: {v}")
              logging.info("-----------------------------------------------")
              # İsteğe bağlı: Yüklenen argümanlarla mevcut argümanları karşılaştır
              # for k, v in args_dict.items():
              #      if k in loaded_args_dict and loaded_args_dict[k] != v:
              #           logging.warning(f"Argument mismatch: '{k}' loaded as {loaded_args_dict[k]}, current is {v}")
         except Exception as e: logging.warning(f"Could not reload config.json: {e}")


    # Rastgele tohumları ayarla (her zaman, devam etse bile determinizm için önemli olabilir)
    # Ancak checkpoint'ten yüklenen state'ler bunu geçersiz kılabilir.
    # Genellikle sadece sıfırdan başlarken ayarlamak daha mantıklıdır.
    if not initial_state_loaded:
        try:
            seed = args.seed
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device.type == 'cuda': torch.cuda.manual_seed_all(seed) # GPU için de
            # Potansiyel olarak deterministik algoritmaları zorla (performansı düşürebilir)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            logging.info(f"Using random seed: {seed}")
        except Exception as e: logging.warning(f"Could not set all random seeds: {e}")


    # Veri Üretimi (her zaman, checkpoint veriyi içermiyorsa)
    # Büyük veri setleri için veriyi kaydet/yükle mekanizması daha iyi olabilir.
    try:
        logging.info("Generating/Reloading data...")
        X_train, y_train = generate_data(args.train_samples, args.seq_length)
        X_test, y_test = generate_data(args.test_samples, args.seq_length)
        input_shape = X_train.shape[1] # Sadece özellik sayısı
        output_shape = y_train.shape[1]
    except Exception:
        logging.critical("Failed to generate/reload data. Exiting.")
        sys.exit(1)


    # Popülasyon Başlatma (sadece sıfırdan başlarken)
    if not initial_state_loaded:
        logging.info(f"--- Initializing Population (Size: {args.pop_size}) ---")
        try:
            population = [create_individual_pytorch(input_shape, output_shape).to(device) for _ in range(args.pop_size)]
            logging.info("Population initialized successfully.")
        except Exception:
            logging.critical("Failed to initialize population. Exiting.")
            sys.exit(1)


    # Evrim Süreci
    logging.info(f"--- Starting/Resuming PyTorch Evolution ({args.generations} Total Generations) ---")
    best_model_evolved: Optional[NeuralNetwork] = None
    best_fitness_hist = loaded_history_best # Yüklenen geçmişle başla
    avg_fitness_hist = loaded_history_avg

    if start_generation >= args.generations:
        logging.warning(f"Loaded checkpoint generation ({start_generation}) is already >= total generations ({args.generations}). Skipping evolution.")
        # Checkpoint'ten en iyi modeli ve geçmişi düzgün yüklemek önemli
        # Şimdilik en iyi modeli popülasyondaki ilk model varsayalım (bu doğru olmayabilir!)
        if population:
             # TODO: Checkpoint'e en iyi modeli de kaydetmek daha iyi olur.
             # Geçici çözüm: Son popülasyondan en iyiyi seç
              try:
                   logging.info("Selecting best model from loaded population as evolution is skipped...")
                   fitness_scores_loaded = [calculate_fitness_pytorch(ind, torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), device) for ind in population]
                   valid_scores_loaded = [(s, i) for i, s in enumerate(fitness_scores_loaded) if np.isfinite(s)]
                   if valid_scores_loaded:
                       best_idx_loaded = max(valid_scores_loaded, key=lambda item: item[0])[1]
                       best_model_evolved = clone_pytorch_model(population[best_idx_loaded], device) # Klonla
                       logging.info(f"Using model {best_model_evolved.model_name} from loaded population as best evolved model.")
                   else:
                       logging.warning("Could not determine best model from loaded population (no finite fitness).")
                       best_model_evolved = None
              except Exception as e:
                   logging.error(f"Error selecting best model from loaded population: {e}")
                   best_model_evolved = None
        else:
             best_model_evolved = None # Popülasyon yüklenememişse
        # Geçmişi de yüklemek lazım (yukarıda TODO olarak belirtildi)
        best_fitness_hist, avg_fitness_hist = [], []
    else:
        try:
            best_model_evolved, gen_best_hist, gen_avg_hist = evolve_population_pytorch(
                population, X_train, y_train, start_generation, args.generations,
                args.crossover_rate, args.mutation_rate, args.weight_mut_rate, args.mutation_strength,
                args.tournament_size, args.elitism_count, args.batch_size, # batch_size evrimde doğrudan kullanılmıyor
                output_dir, args.checkpoint_interval, device
            )
            # Yüklenen geçmişle bu çalıştırmanın geçmişini birleştir
            best_fitness_hist.extend(gen_best_hist)
            avg_fitness_hist.extend(gen_avg_hist)

        except Exception as e:
            logging.critical(f"Fatal error during PyTorch evolution process: {e}", exc_info=True)
            sys.exit(1)
    logging.info("--- PyTorch Evolution Complete ---")

    # Fitness geçmişini kaydetme ve çizdirme
    if best_fitness_hist or avg_fitness_hist:
        plot_fitness_history(best_fitness_hist, avg_fitness_hist, output_dir)
        history_path = os.path.join(output_dir, "fitness_history_pytorch.csv")
        try:
            # Geçmişi CSV olarak kaydet
            history_data = np.array([
                np.arange(1, len(best_fitness_hist) + 1), # Nesil numaraları (1'den başlayarak)
                best_fitness_hist,
                avg_fitness_hist
            ]).T
            np.savetxt(history_path, history_data, delimiter=',', header='Generation,BestFitness,AvgFitness', comments='', fmt=['%d', '%.8f', '%.8f'])
            logging.info(f"Full fitness history saved to {history_path}")
        except Exception as e:
            logging.error(f"Could not save fitness history data: {e}")
    else:
        logging.warning("Fitness history is empty after evolution, skipping saving/plotting.")


    # En iyi modelin son eğitimi, değerlendirme ve sonuç kaydı
    final_model_path = None
    training_summary = {}
    final_metrics = {"test_mse": np.inf, "avg_kendall_tau": 0.0}
    best_model_architecture = {}

    if best_model_evolved is None:
        logging.error("Evolution did not yield a best model. Skipping final training and evaluation.")
    else:
        best_model_architecture = best_model_evolved.get_architecture()
        logging.info(f"Best evolved model architecture: {best_model_architecture}")
        # Model özetini logla (parametre sayısı vb.)
        try:
            num_params = sum(p.numel() for p in best_model_evolved.parameters() if p.requires_grad)
            logging.info(f"Best Evolved Model ({best_model_evolved.model_name}) - Trainable Parameters: {num_params}")
            # Daha detaylı özet için torchinfo gibi kütüphaneler kullanılabilir:
            # from torchinfo import summary
            # summary(best_model_evolved, input_size=(args.batch_size, input_shape)) # input_size örnektir
        except Exception as e:
            logging.warning(f"Could not log model summary details: {e}")


        # Son Eğitim
        try:
             # Eğitmeden önce bir klonunu alalım ki orijinal evrimleşmiş hali kaybolmasın
             model_to_train = clone_pytorch_model(best_model_evolved, device)
             final_model, training_summary = train_final_model_pytorch(
                 model_to_train, X_train, y_train,
                 args.epochs_final_train, args.batch_size, args.learning_rate, # Args'a learning_rate ekle
                 device, output_dir
             )
        except Exception as e:
             logging.error(f"Error during final training setup or execution: {e}", exc_info=True)
             final_model = None # Eğitim başarısız
             training_summary = {"error": str(e)}

        # Değerlendirme
        if final_model:
             final_metrics = evaluate_model_pytorch(final_model, X_test, y_test, args.batch_size, device)
             # Son eğitilmiş modeli kaydet
             final_model_path = os.path.join(output_dir, "best_evolved_model_trained_pytorch.pt")
             try:
                 # Sadece state_dict kaydetmek genellikle daha iyidir
                 torch.save({
                     'architecture': final_model.get_architecture(),
                     'model_state_dict': final_model.state_dict(),
                     # 'optimizer_state_dict': optimizer.state_dict(), # Eğitimde kullanılan optimizatör durumu
                     'training_summary': training_summary,
                     'evaluation_metrics': final_metrics
                 }, final_model_path)
                 logging.info(f"Final trained model state and architecture saved to {final_model_path}")
             except Exception as e:
                 logging.error(f"Failed to save final trained model: {e}", exc_info=True)
                 final_model_path = None # Kaydedilemedi
        else:
             logging.error("Final model training failed or did not produce a model. Skipping evaluation and saving.")


    logging.info("--- Saving Final Results ---")
    final_results = {
        "run_info": {
            "run_name": run_name,
            "timestamp": timestamp,
            "output_directory": output_dir,
            "framework": "PyTorch",
            "device_used": str(device),
            "resumed_run": resume_run,
            "last_checkpoint_loaded": latest_checkpoint_path
        },
        "config": args_dict, # Başlangıç argümanları
        "evolution_summary": {
            "start_generation": start_generation,
            "end_generation": start_generation + len(best_fitness_hist) - (1 if loaded_history_best else 0), # Çalıştırılan son nesil
            "generations_run_this_session": len(best_fitness_hist) - len(loaded_history_best),
            "best_fitness_achieved_overall": max(best_fitness_hist) if best_fitness_hist and any(np.isfinite(f) for f in best_fitness_hist) else None,
            "best_fitness_final_gen": best_fitness_hist[-1] if best_fitness_hist and np.isfinite(best_fitness_hist[-1]) else None,
            "avg_fitness_final_gen": avg_fitness_hist[-1] if avg_fitness_hist and np.isfinite(avg_fitness_hist[-1]) else None,
            "best_model_architecture": best_model_architecture
        },
        "final_training_summary": training_summary,
        "final_evaluation_on_test": final_metrics,
        "saved_trained_model_path": final_model_path
    }
    results_path = os.path.join(output_dir, "final_results_pytorch.json")
    try:
        # NumPy ve diğer serileştirilemeyen türleri JSON'a uygun hale getir
        def convert_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, torch.Tensor): return obj.tolist() # Tensörleri listeye çevir
            elif isinstance(obj, torch.device): return str(obj) # Cihazı string yap
            elif isinstance(obj, type): return obj.__name__ # Türleri isim olarak kaydet
            return obj
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4, default=convert_types, sort_keys=True)
        logging.info(f"Final results summary saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save final results JSON: {e}", exc_info=True)

    logging.info(f"========== PyTorch Pipeline Run {run_name} Finished ==========")


# --- Argüman Ayrıştırıcı (PyTorch için Eklemeler) ---
def parse_arguments_v4() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoNet v4: Neuroevolution with PyTorch, Crossover & Checkpointing")

    # --- Dizinler ve Kontrol ---
    parser.add_argument('--output_base_dir', type=str, default=DEFAULT_OUTPUT_BASE_DIR, help='Base directory for new runs.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a previous run directory to resume from (PyTorch checkpoints).')
    parser.add_argument('--checkpoint_interval', type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help='Save checkpoint every N generations (0 to disable).')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=['auto', 'cpu', 'cuda'], help='Device to use (cpu, cuda, or auto-detect).')

    # --- Veri Ayarları ---
    parser.add_argument('--seq_length', type=int, default=DEFAULT_SEQ_LENGTH, help='Length of sequences.')
    parser.add_argument('--train_samples', type=int, default=5000, help='Number of training samples.')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples.')

    # --- Evrim Parametreleri ---
    parser.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE, help='Population size.')
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS, help='Total number of generations.')
    parser.add_argument('--crossover_rate', type=float, default=DEFAULT_CROSSOVER_RATE, help='Probability of applying crossover.')
    parser.add_argument('--mutation_rate', type=float, default=DEFAULT_MUTATION_RATE, help='Probability of applying mutation (if crossover is not applied).')
    parser.add_argument('--weight_mut_rate', type=float, default=DEFAULT_WEIGHT_MUT_RATE, help='Probability for each weight/bias to be mutated if mutation occurs.')
    parser.add_argument('--mutation_strength', type=float, default=DEFAULT_MUTATION_STRENGTH, help='Std dev for weight mutation noise (Gaussian).')
    parser.add_argument('--tournament_size', type=int, default=DEFAULT_TOURNAMENT_SIZE, help='Tournament selection size.')
    parser.add_argument('--elitism_count', type=int, default=DEFAULT_ELITISM_COUNT, help='Number of elite individuals to carry over.')

    # --- Eğitim ve Değerlendirme ---
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for final training and evaluation.')
    parser.add_argument('--epochs_final_train', type=int, default=DEFAULT_EPOCHS_FINAL_TRAIN, help='Max epochs for final training of the best model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer during final training.')

    # --- Tekrarlanabilirlik ---
    parser.add_argument('--seed', type=int, default=None, help='Random seed for Python, NumPy, and PyTorch (default: random).')

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {args.seed}")

    # Basit kontroller
    if args.elitism_count >= args.pop_size:
        print(f"Warning: Elitism count ({args.elitism_count}) >= Population size ({args.pop_size}). Setting elitism to PopSize - 1.")
        args.elitism_count = max(0, args.pop_size - 1)
    if args.tournament_size <= 0:
         print(f"Warning: Tournament size ({args.tournament_size}) must be > 0. Setting to 1.")
         args.tournament_size = 1
    if args.tournament_size > args.pop_size:
         print(f"Warning: Tournament size ({args.tournament_size}) > Population size ({args.pop_size}). Setting to PopSize.")
         args.tournament_size = args.pop_size

    return args


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    cli_args = parse_arguments_v4()
    try:
        run_pipeline_pytorch(cli_args)
    except SystemExit:
        logging.info("SystemExit caught, exiting gracefully.")
        pass # Argparse veya bilinçli çıkışlar için
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting...")
        logging.warning("KeyboardInterrupt detected. Attempting graceful shutdown.")
        sys.exit(130) # Ctrl+C için standart çıkış kodu
    except Exception as e:
        # Loglama zaten ayarlandıysa, kritik hata logla
        if logging.getLogger().hasHandlers():
            logging.critical("FATAL UNHANDLED ERROR in main execution block:", exc_info=True)
        else: # Loglama başlamadan hata olursa stderr'a yaz
            import traceback
            print(f"\nFATAL UNHANDLED ERROR in main execution block: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1) # Başarısız çıkış kodu