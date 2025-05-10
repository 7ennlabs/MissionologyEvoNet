# ==============================================================================
# EvoNet Optimizer - v5 - Adaptif & Paralel PyTorch Sürümü
# Açıklama: v4 üzerine inşa edilmiştir. Adaptif mutasyon gücü, fitness'ta
#           karmaşıklık cezası, paralel fitness hesaplama (CPU),
#           opsiyonel Weights & Biases entegrasyonu ve genel iyileştirmeler içerir.
# ==============================================================================

import os
# os.environ["WANDB_SILENT"] = "true" # W&B loglarını azaltmak için (isteğe bağlı)
import sys
import argparse
import random
import logging
from datetime import datetime
import json
import copy
import time
from typing import List, Tuple, Dict, Any, Optional, Union
import concurrent.futures # Paralel fitness hesaplama için

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Opsiyonel W&B importu
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    print("Warning: wandb library not found. Experiment tracking with W&B is disabled.")
    print("Install with: pip install wandb")


# --- Sabitler ve Varsayılan Değerler ---
DEFAULT_SEQ_LENGTH = 10
DEFAULT_POP_SIZE = 50
DEFAULT_GENERATIONS = 50
DEFAULT_CROSSOVER_RATE = 0.6
DEFAULT_MUTATION_RATE = 0.4
DEFAULT_WEIGHT_MUT_RATE = 0.8
DEFAULT_MUTATION_STRENGTH = 0.1 # Başlangıç mutasyon gücü
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_ELITISM_COUNT = 2
DEFAULT_EPOCHS_FINAL_TRAIN = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "evonet_runs_v5_pytorch")
DEFAULT_CHECKPOINT_INTERVAL = 10
DEFAULT_DEVICE = "auto"
DEFAULT_NUM_WORKERS = 0 # Paralel fitness için worker sayısı (0 = Kapalı/Ana thread)

# Adaptif Mutasyon Parametreleri
DEFAULT_ADAPT_MUTATION = True
DEFAULT_STAGNATION_LIMIT = 10 # İyileşme olmazsa adaptasyon için nesil sayısı
DEFAULT_MUT_STRENGTH_DECAY = 0.98 # İyileşme olduğunda azaltma faktörü
DEFAULT_MUT_STRENGTH_INCREASE = 1.1 # Tıkanma olduğunda artırma faktörü
DEFAULT_MIN_MUT_STRENGTH = 0.005
DEFAULT_MAX_MUT_STRENGTH = 0.5

# Gelişmiş Fitness Parametreleri
DEFAULT_COMPLEXITY_PENALTY = 0.00001 # Parametre başına ceza ağırlığı


# --- Loglama Ayarları ---
# (setup_logging fonksiyonu öncekiyle aynı, v4'teki gibi)
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    log_filename = os.path.join(log_dir, 'evolution_run_pytorch_v5.log')
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-8s [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("="*50)
    logging.info("PyTorch EvoNet v5 Logging Başlatıldı.")
    logging.info("="*50)

# --- Cihaz (GPU/CPU) Ayarları ---
# (setup_device fonksiyonu öncekiyle aynı, v4'teki gibi)
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
# (generate_data fonksiyonu öncekiyle aynı, v4'teki gibi)
def generate_data(num_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    logging.info(f"Generating {num_samples} samples with sequence length {seq_length}...")
    try:
        X = np.random.rand(num_samples, seq_length).astype(np.float32) * 100
        y = np.sort(X, axis=1).astype(np.float32)
        logging.info("Data generation successful.")
        return X, y
    except Exception as e:
        logging.error(f"Error during data generation: {e}", exc_info=True)
        raise

# --- PyTorch Sinir Ağı Modeli ---
# (NeuralNetwork sınıfı öncekiyle büyük ölçüde aynı, v4'teki gibi)
# Küçük iyileştirme: get_num_params metodu eklendi.
class NeuralNetwork(nn.Module):
    """ Dinamik olarak yapılandırılabilen basit bir PyTorch MLP modeli. """
    def __init__(self, input_size: int, output_size: int, hidden_dims: List[int], activations: List[str]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims
        self.activations_str = activations

        layers = []
        last_dim = input_size
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(last_dim, h_dim))
            act_func_str = activations[i].lower()
            if act_func_str == 'relu': layers.append(nn.ReLU())
            elif act_func_str == 'tanh': layers.append(nn.Tanh())
            elif act_func_str == 'sigmoid': layers.append(nn.Sigmoid())
            else:
                logging.warning(f"Bilinmeyen aktivasyon '{activations[i]}', ReLU kullanılıyor.")
                layers.append(nn.ReLU())
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, output_size))

        self.network = nn.Sequential(*layers)
        self.architecture_id = self._generate_architecture_id()
        self.model_name = f"model_{self.architecture_id}_rnd{random.randint(10000, 99999)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_architecture(self) -> Dict[str, Any]:
        return {"input_size": self.input_size, "output_size": self.output_size,
                "hidden_dims": self.hidden_dims, "activations": self.activations_str}

    def _generate_architecture_id(self) -> str:
        h_dims_str = '_'.join(map(str, self.hidden_dims))
        acts_str = ''.join([a[0].upper() for a in self.activations_str])
        return f"I{self.input_size}_H{h_dims_str}_A{acts_str}_O{self.output_size}"

    def get_num_params(self, trainable_only: bool = True) -> int:
        """ Modeldeki parametre sayısını döndürür. """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def __eq__(self, other):
        if not isinstance(other, NeuralNetwork): return NotImplemented
        return self.get_architecture() == other.get_architecture()

    def __hash__(self):
         arch_tuple = (self.input_size, self.output_size, tuple(self.hidden_dims), tuple(self.activations_str))
         return hash(arch_tuple)


# --- Neuroevolution Çekirdeği (PyTorch v5) ---

# (create_individual_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
def create_individual_pytorch(input_size: int, output_size: int) -> NeuralNetwork:
    """ Rastgele mimariye sahip bir PyTorch NeuralNetwork modeli oluşturur. """
    try:
        num_hidden_layers = random.randint(1, 4)
        hidden_dims = [random.randint(16, 128) for _ in range(num_hidden_layers)]
        activations = [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(num_hidden_layers)]
        model = NeuralNetwork(input_size, output_size, hidden_dims, activations)
        logging.debug(f"Created individual: {model.model_name} with {model.get_num_params()} params")
        return model
    except Exception as e:
        logging.error(f"Error creating PyTorch individual model: {e}", exc_info=True)
        raise

# (clone_pytorch_model fonksiyonu öncekiyle aynı, v4'teki gibi)
def clone_pytorch_model(model: NeuralNetwork, device: torch.device) -> NeuralNetwork:
    """ Bir PyTorch modelini (mimari ve ağırlıklar) klonlar. """
    try:
        arch = model.get_architecture()
        cloned_model = NeuralNetwork(**arch)
        cloned_model.load_state_dict(copy.deepcopy(model.state_dict()))
        cloned_model.to(device)
        cloned_model.model_name = f"cloned_{model.model_name}_{random.randint(1000,9999)}"
        logging.debug(f"Cloned model {model.model_name} to {cloned_model.model_name}")
        return cloned_model
    except Exception as e:
        logging.error(f"Error cloning PyTorch model {model.model_name}: {e}", exc_info=True)
        raise


# Bu fonksiyon paralel işçiler tarafından çağrılacak
# Doğrudan model objesi yerine state_dict ve mimari alıyor
def _calculate_fitness_worker(
    model_arch: Dict[str, Any],
    model_state_dict: Dict[str, torch.Tensor],
    X_np: np.ndarray, # Veriyi NumPy olarak alalım
    y_np: np.ndarray,
    device_str: str, # Cihazı string olarak alalım
    fitness_params: Dict # Karmaşıklık cezası vb.
) -> float:
    """ Bir modelin fitness'ını hesaplayan işçi fonksiyonu (paralel kullanım için). """
    try:
        # 1. Modeli yeniden oluştur
        device = torch.device(device_str)
        model = NeuralNetwork(**model_arch)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        # 2. Veriyi Tensör'e çevir ve cihaza taşı
        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)

        # 3. Fitness Hesaplama (v4'teki calculate_fitness_pytorch benzeri)
        complexity_penalty_weight = fitness_params.get('complexity_penalty', 0.0)

        with torch.no_grad():
            y_pred = model(X)
            mse_val = torch.mean((y_pred - y)**2).item()

        if not np.isfinite(mse_val):
            # Worker'da loglama yapmak yerine None veya özel bir değer döndürebiliriz
            # logging.warning(f"Worker: Non-finite MSE ({mse_val}) for model.")
            return -np.inf # Ana süreçte işlenecek

        # Temel fitness (MSE'nin tersi)
        fitness_score = 1.0 / (mse_val + 1e-9)

        # Karmaşıklık Cezası Ekle
        if complexity_penalty_weight > 0:
            num_params = model.get_num_params(trainable_only=True)
            complexity_penalty = complexity_penalty_weight * num_params
            fitness_score -= complexity_penalty
            # print(f"Debug: Model params: {num_params}, penalty: {complexity_penalty:.4f}, score after penalty: {fitness_score:.4f}") # DEBUG

        # --- KAVRAMSAL: Diğer Gelişmiş Fitness Metrikleri ---
        # tau_weight = fitness_params.get('w_tau', 0.0)
        # if tau_weight > 0:
        #     y_np_local = y.cpu().numpy()
        #     y_pred_np_local = y_pred.cpu().numpy()
        #     tau_val = calculate_avg_kendall_tau(y_np_local, y_pred_np_local, sample_size=100)
        #     fitness_score += tau_weight * tau_val
        # ----------------------------------------------------

        if not np.isfinite(fitness_score):
            return -np.inf

        return float(fitness_score)

    except Exception as e:
        # Hataları ana sürece bildirmek için loglama yerine None/exception döndürmek daha iyi olabilir
        # Ancak basitlik için burada loglayıp çok düşük değer döndürelim
        # logging.error(f"Error in fitness worker: {e}", exc_info=True) # Bu log dosyasına yazılmaz
        print(f"[Worker Error] Failed to calculate fitness: {e}", file=sys.stderr) # stderr'a yazdır
        return -np.inf # Hata durumunda


# (mutate_individual_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
# Sadece mutasyon gücünü parametre olarak alıyor
def mutate_individual_pytorch(
    individual: NeuralNetwork,
    weight_mut_rate: float,
    current_mutation_strength: float, # Adaptif olarak gelen güç
    device: torch.device
) -> NeuralNetwork:
    """ Bir PyTorch bireyine adaptif güçle ağırlık mutasyonu uygular. """
    try:
        mutated_model = clone_pytorch_model(individual, device)
        mutated_model.model_name = f"mutated_{individual.model_name}_{random.randint(1000,9999)}"
        mutated = False
        state_dict = mutated_model.state_dict()
        new_state_dict = copy.deepcopy(state_dict)

        for name, param in new_state_dict.items():
            if param.requires_grad and random.random() < weight_mut_rate :
                 mutated = True
                 noise = torch.randn_like(param) * current_mutation_strength # Adaptif gücü kullan
                 new_state_dict[name] = param + noise.to(param.device)

        if mutated:
            mutated_model.load_state_dict(new_state_dict)
            logging.debug(f"Mutated model {individual.model_name} -> {mutated_model.model_name} with strength {current_mutation_strength:.4f}")
            return mutated_model
        else:
            logging.debug(f"Mutation applied to {individual.model_name}, but no weights changed based on rate.")
            return mutated_model # Klonlanmış modeli döndür

    except Exception as e:
        logging.error(f"Error during PyTorch mutation of model {individual.model_name}: {e}", exc_info=True)
        return clone_pytorch_model(individual, device)


# (check_architecture_compatibility_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
def check_architecture_compatibility_pytorch(model1: NeuralNetwork, model2: NeuralNetwork) -> bool:
    return model1.get_architecture() == model2.get_architecture()

# (crossover_individuals_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
def crossover_individuals_pytorch(
    parent1: NeuralNetwork,
    parent2: NeuralNetwork,
    device: torch.device
) -> Tuple[Optional[NeuralNetwork], Optional[NeuralNetwork]]:
    """ İki PyTorch ebeveynden basit ağırlık ortalaması/karıştırması ile çocuklar oluşturur. """
    if not check_architecture_compatibility_pytorch(parent1, parent2):
        logging.debug(f"Skipping crossover between {parent1.model_name} and {parent2.model_name} due to incompatible architectures.")
        return None, None
    try:
        arch = parent1.get_architecture()
        child1 = NeuralNetwork(**arch).to(device)
        child2 = NeuralNetwork(**arch).to(device)
        child1.model_name = f"xover_{parent1.architecture_id}_c1_{random.randint(1000,9999)}"
        child2.model_name = f"xover_{parent1.architecture_id}_c2_{random.randint(1000,9999)}"
        p1_state, p2_state = parent1.state_dict(), parent2.state_dict()
        c1_state, c2_state = child1.state_dict(), child2.state_dict()
        for name in p1_state:
            param1, param2 = p1_state[name], p2_state[name]
            mask = torch.rand_like(param1) < 0.5
            c1_state[name] = torch.where(mask, param1, param2)
            c2_state[name] = torch.where(mask, param2, param1)
        child1.load_state_dict(c1_state)
        child2.load_state_dict(c2_state)
        logging.debug(f"Crossover performed between {parent1.model_name} and {parent2.model_name}")
        return child1, child2
    except Exception as e:
        logging.error(f"Error during PyTorch crossover between {parent1.model_name} and {parent2.model_name}: {e}", exc_info=True)
        return None, None

# (tournament_selection fonksiyonu öncekiyle aynı, v4'teki gibi)
def tournament_selection(
    population: List[NeuralNetwork],
    fitness_scores: List[float],
    k: int
) -> NeuralNetwork:
    """ Turnuva seçimi ile popülasyondan bir birey seçer. """
    if not population: raise ValueError("Population cannot be empty")
    valid_indices = [i for i, score in enumerate(fitness_scores) if np.isfinite(score)]
    if not valid_indices:
        logging.warning("No individuals with finite fitness scores found for tournament selection. Returning random individual.")
        return random.choice(population)
    if len(valid_indices) < k: k = len(valid_indices)
    if k <= 0: k = 1

    try:
        # Sadece geçerli fitness'a sahip olanlar arasından seç
        tournament_indices_pool = random.sample(valid_indices, k)
        tournament_contenders = [(fitness_scores[i], population[i]) for i in tournament_indices_pool]
        winner = max(tournament_contenders, key=lambda item: item[0])[1]
        return winner
    except Exception as e:
        logging.error(f"Error during tournament selection: {e}", exc_info=True)
        return random.choice(population) # Hata durumunda rastgele


# --- Checkpointing (PyTorch v5) ---
# (save_checkpoint_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
# İsteğe bağlı: Adaptif durum veya W&B run ID'si eklenebilir.
def save_checkpoint_pytorch(output_dir: str, generation: int, population: List[NeuralNetwork],
                            rnd_state: Any, np_rnd_state: Any, torch_rnd_state: Any,
                            wandb_run_id: Optional[str] = None): # W&B ID'si ekle
    """ Evrim durumunu (PyTorch v5) kaydeder. """
    checkpoint_dir = os.path.join(output_dir, "checkpoints_pytorch_v5")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"evo_gen_{generation}.pt")
    logging.info(f"Saving checkpoint for generation {generation} to {checkpoint_file}...")

    population_state = []
    for model in population:
        try:
            population_state.append({
                "name": model.model_name,
                "architecture": model.get_architecture(),
                "state_dict": model.state_dict()
            })
        except Exception as e:
            logging.error(f"Could not serialize model {model.model_name} for checkpoint: {e}")

    state = {
        "version": "v5", # Sürüm bilgisi ekle
        "generation": generation,
        "population_state": population_state,
        "random_state": rnd_state,
        "numpy_random_state": np_rnd_state,
        "torch_random_state": torch_rnd_state,
        "wandb_run_id": wandb_run_id, # W&B run ID
        "timestamp": datetime.now().isoformat()
        # İsteğe bağlı: Adaptif mutasyonun mevcut durumu (current_mutation_strength, stagnation_counter)
    }
    try:
        torch.save(state, checkpoint_file)
        logging.info(f"Checkpoint saved successfully for generation {generation}.")
    except Exception as e:
        logging.error(f"Failed to save checkpoint using torch.save for generation {generation}: {e}", exc_info=True)

# (load_checkpoint_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
# Sadece W&B run ID'sini okur
def load_checkpoint_pytorch(checkpoint_path: str, device: torch.device) -> Optional[Dict]:
    """ Kaydedilmiş PyTorch v5 evrim durumunu yükler. """
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        return None
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if checkpoint.get("version") != "v5":
             logging.warning(f"Loading checkpoint from a different version ({checkpoint.get('version', 'Unknown')}). Compatibility not guaranteed.")

        population = []
        for model_state in checkpoint["population_state"]:
            try:
                arch = model_state["architecture"]
                model = NeuralNetwork(**arch)
                model.load_state_dict(model_state["state_dict"])
                model.to(device)
                model.model_name = model_state.get("name", f"loaded_model_{random.randint(1000,9999)}")
                model.eval()
                population.append(model)
            except Exception as e:
                logging.error(f"Failed to load model state from checkpoint for model {model_state.get('name', 'UNKNOWN')}: {e}", exc_info=True)

        if not population:
            logging.error("Failed to load any model from the checkpoint population state.")
            return None

        checkpoint["population"] = population
        logging.info(f"Checkpoint loaded successfully. Resuming from generation {checkpoint['generation'] + 1}.")
        # W&B ID'sini döndür
        checkpoint["wandb_run_id"] = checkpoint.get("wandb_run_id")
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
        return None

# (find_latest_checkpoint_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
# Sadece klasör adını v5'e göre güncelleyebiliriz
def find_latest_checkpoint_pytorch(output_dir: str) -> Optional[str]:
    """ Verilen klasördeki en son PyTorch v5 checkpoint dosyasını (.pt) bulur. """
    checkpoint_dir = os.path.join(output_dir, "checkpoints_pytorch_v5") # v5 klasörü
    if not os.path.isdir(checkpoint_dir): return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("evo_gen_") and f.endswith(".pt")]
    if not checkpoints: return None
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


# --- Ana Evrim Döngüsü (PyTorch v5 - Adaptif, Paralel) ---
def evolve_population_pytorch_v5(
    population: List[NeuralNetwork],
    X_train_np: np.ndarray, y_train_np: np.ndarray, # Veriyi NumPy olarak al
    start_generation: int, total_generations: int,
    crossover_rate: float, mutation_rate: float, weight_mut_rate: float,
    args: argparse.Namespace, # Tüm argümanları alalım
    output_dir: str, device: torch.device,
    wandb_run: Optional[Any] # W&B run objesi
) -> Tuple[Optional[NeuralNetwork], List[float], List[float]]:
    """ PyTorch v5 tabanlı evrimsel süreci çalıştırır (Adaptif, Paralel). """

    best_fitness_history = []
    avg_fitness_history = []
    best_model_overall: Optional[NeuralNetwork] = None
    best_fitness_overall = -np.inf

    # Adaptif Mutasyon için başlangıç değerleri
    current_mutation_strength = args.mutation_strength
    stagnation_counter = 0

    pop_size = len(population)
    fitness_params = {'complexity_penalty': args.complexity_penalty} # Fitness worker için parametreler

    # Paralel işleyici havuzu (eğer worker > 0 ise)
    # 'fork' yerine 'spawn' kullanmak daha güvenli olabilir (özellikle CUDA ile)
    # Ancak 'spawn' daha fazla overhead yaratabilir. Duruma göre seçilebilir.
    # context = torch.multiprocessing.get_context("spawn") if args.num_workers > 0 else None
    # executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=context) if args.num_workers > 0 else None
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) if args.num_workers > 0 else None
    if executor:
         logging.info(f"Using ProcessPoolExecutor with {args.num_workers} workers for fitness evaluation.")

    try: # Executor'ı düzgün kapatmak için try...finally
        for gen in range(start_generation, total_generations):
            generation_start_time = time.time()

            # 1. Fitness Değerlendirme (Paralel veya Seri)
            fitness_scores = [-np.inf] * pop_size # Başlangıç değeri
            population_states = [(ind.get_architecture(), ind.state_dict()) for ind in population]

            try:
                if executor and args.num_workers > 0:
                    futures = [executor.submit(_calculate_fitness_worker,
                                               arch, state, X_train_np, y_train_np,
                                               str(device), fitness_params)
                               for arch, state in population_states]
                    # concurrent.futures.wait(futures) # Beklemeye gerek yok, as_completed daha iyi
                    results = []
                    # Sonuçları geldikçe işle (sırasız gelebilir)
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result = future.result()
                            results.append(result)
                            # print(f"DEBUG: Worker {i} finished with fitness {result}") # DEBUG
                        except Exception as exc:
                            logging.error(f"Fitness calculation job {i} generated an exception: {exc}")
                            results.append(-np.inf) # Hata durumunda minimum fitness
                    # Sonuçları doğru sıraya koymak GEREKLİ DEĞİL çünkü seçilim/elitizm zaten skorlara göre çalışır
                    # Ancak loglama/takip için orijinal sıra önemliyse, future'ları dict ile takip edip sıraya dizmek gerekir.
                    # Basitlik için, sonuç listesinin popülasyonla aynı sırada olduğunu varsayalım (as_completed sırayı bozar!)
                    # DÜZELTME: Sonuçları sıraya dizmek ŞART. Future'ları indeksle takip et.
                    results_map = {}
                    futures_map = {executor.submit(_calculate_fitness_worker,
                                                   pop[0], pop[1], X_train_np, y_train_np,
                                                   str(device), fitness_params): index
                                   for index, pop in enumerate(population_states)}

                    for future in concurrent.futures.as_completed(futures_map):
                        original_index = futures_map[future]
                        try:
                            result = future.result()
                            fitness_scores[original_index] = result
                        except Exception as exc:
                            logging.error(f'Individual {original_index} generated an exception: {exc}')
                            fitness_scores[original_index] = -np.inf # Hata durumunda

                else: # Seri hesaplama (num_workers=0)
                    logging.debug("Calculating fitness sequentially...")
                    temp_device = torch.device("cpu") # Seri hesaplamayı CPU'da yapmak GPU'yu meşgul etmez
                    # Ana süreçte modeli CPU'ya taşı, hesapla, sonucu al
                    for i, (arch, state) in enumerate(population_states):
                         # Modeli her seferinde yeniden oluşturmak yerine klonlamak daha verimli olabilir mi?
                         # Ancak _calculate_fitness_worker mantığına uymak için yeniden oluşturalım.
                         try:
                              model_instance = NeuralNetwork(**arch)
                              model_instance.load_state_dict(state)
                              model_instance.to(temp_device)
                              fitness_scores[i] = calculate_fitness_pytorch( # Bu fonksiyon artık sadece seri için
                                                                           model_instance, X_train_np, y_train_np,
                                                                           temp_device, fitness_params)
                         except Exception as e:
                              logging.error(f"Error calculating fitness for individual {i} sequentially: {e}")
                              fitness_scores[i] = -np.inf


            except Exception as e:
                logging.critical(f"Error during fitness evaluation distribution/collection in Gen {gen+1}: {e}", exc_info=True)
                raise # Bu kritik bir hata, devam etmek zor

            # Fitness hesaplama sonrası GPU belleğini temizle (paralel workerlar ayrı process olduğu için burada etkisi olmaz ama seri için kalabilir)
            # if device.type == 'cuda': torch.cuda.empty_cache()

            # 2. İstatistikler ve En İyiyi Takip
            valid_indices = [i for i, score in enumerate(fitness_scores) if np.isfinite(score)]
            if not valid_indices:
                logging.error(f"Generation {gen+1}: No individuals with finite fitness scores found! Cannot proceed.")
                # Burada ne yapmalı? Popülasyonu sıfırlamak mı, durmak mı? Şimdilik duralım.
                raise RuntimeError(f"Evolution stopped at generation {gen+1} due to lack of valid individuals.")

            current_best_idx_local = np.argmax([fitness_scores[i] for i in valid_indices])
            current_best_idx_global = valid_indices[current_best_idx_local]
            current_best_fitness = fitness_scores[current_best_idx_global]

            finite_scores = [fitness_scores[i] for i in valid_indices]
            avg_fitness = np.mean(finite_scores)

            best_fitness_history.append(current_best_fitness)
            avg_fitness_history.append(avg_fitness)

            new_best_found = False
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                new_best_found = True
                try:
                    best_model_overall = clone_pytorch_model(population[current_best_idx_global], device)
                    logging.info(f"Generation {gen+1}: *** New overall best fitness: {best_fitness_overall:.6f} (Model: {best_model_overall.model_name}) ***")
                except Exception as e:
                    logging.error(f"Could not clone new best model {population[current_best_idx_global].model_name}: {e}", exc_info=True)
                    best_model_overall = None
            # else: # En iyi bulunamadıysa veya aynıysa
            #     pass # Stagnation sayacı aşağıda artacak

            generation_time = time.time() - generation_start_time
            logging.info(f"Generation {gen+1}/{total_generations} | Best Fitness: {current_best_fitness:.6f} | Avg Fitness: {avg_fitness:.6f} | Mut Strength: {current_mutation_strength:.4f} | Time: {generation_time:.2f}s")

            # W&B Loglama (eğer aktifse)
            if wandb_run:
                try:
                    wandb_run.log({
                        "generation": gen + 1,
                        "best_fitness": current_best_fitness,
                        "average_fitness": avg_fitness,
                        "mutation_strength": current_mutation_strength,
                        "generation_time_sec": generation_time,
                        "num_valid_individuals": len(valid_indices),
                     #   "best_model_params": best_model_overall.get_num_params() if best_model_overall else None # En iyinin parametre sayısı
                    }, step=gen + 1) # Adım olarak nesil numarasını kullan
                except Exception as e:
                    logging.warning(f"Failed to log metrics to W&B: {e}")


            # Adaptif Mutasyon Gücü Güncelleme
            if args.adapt_mutation:
                if new_best_found:
                    stagnation_counter = 0
                    current_mutation_strength = max(args.min_mut_strength, current_mutation_strength * args.mut_strength_decay)
                    logging.debug(f"Improvement found. Decreasing mutation strength to {current_mutation_strength:.4f}")
                else:
                    stagnation_counter += 1
                    logging.debug(f"No improvement. Stagnation counter: {stagnation_counter}")
                    if stagnation_counter >= args.stagnation_limit:
                        current_mutation_strength = min(args.max_mut_strength, current_mutation_strength * args.mut_strength_increase)
                        logging.info(f"Stagnation detected ({stagnation_counter} gens). Increasing mutation strength to {current_mutation_strength:.4f}")
                        stagnation_counter = 0 # Sayacı sıfırla

            # 3. Yeni Popülasyon Oluşturma (Elitizm, Çaprazlama, Mutasyon)
            new_population = []

            # 3a. Elitizm
            if args.elitism_count > 0 and len(population) >= args.elitism_count:
                try:
                    # Sadece geçerli fitness'a sahip elitleri seç
                    sorted_valid_indices = sorted(valid_indices, key=lambda i: fitness_scores[i], reverse=True)
                    elite_indices = sorted_valid_indices[:args.elitism_count]
                    for idx in elite_indices:
                        elite_clone = clone_pytorch_model(population[idx], device)
                        elite_clone.model_name = f"elite_{population[idx].model_name}"
                        new_population.append(elite_clone)
                    logging.debug(f"Added {len(new_population)} elites to the next generation.")
                except Exception as e:
                    logging.error(f"Error during elitism: {e}", exc_info=True)

            # 3b. Kalanları Üretme
            num_to_generate = pop_size - len(new_population)
            generated_count = 0
            reproduction_attempts = 0
            max_reproduction_attempts = num_to_generate * 5 # Daha cömert sınır

            while generated_count < num_to_generate and reproduction_attempts < max_reproduction_attempts:
                reproduction_attempts += 1
                try:
                    parent1 = tournament_selection(population, fitness_scores, args.tournament_size)
                    parent2 = tournament_selection(population, fitness_scores, args.tournament_size)
                    child1, child2 = None, None

                    if random.random() < crossover_rate and parent1 is not parent2:
                        child1, child2 = crossover_individuals_pytorch(parent1, parent2, device)

                    if child1 is None: # Çaprazlama olmadıysa veya başarısızsa
                        if random.random() < mutation_rate:
                             parent_to_mutate = parent1
                             child1 = mutate_individual_pytorch(parent_to_mutate, weight_mut_rate, current_mutation_strength, device)
                        else: # Klonlama
                             child1 = clone_pytorch_model(parent1, device)
                             child1.model_name = f"direct_clone_{parent1.model_name}_{random.randint(1000,9999)}"

                    if child1:
                        new_population.append(child1); generated_count += 1
                        if generated_count >= num_to_generate: break
                    if child2:
                        new_population.append(child2); generated_count += 1
                        if generated_count >= num_to_generate: break

                except Exception as e:
                    logging.error(f"Error during selection/reproduction cycle (attempt {reproduction_attempts}): {e}", exc_info=True)

            if generated_count < num_to_generate:
                logging.warning(f"Reproduction cycle failed to generate enough individuals. Adding {num_to_generate - generated_count} random individuals.")
                # Rastgele bireyleri eklemeden önce popülasyonun boş olmadığından emin ol
                if population:
                     input_s = population[0].input_size
                     output_s = population[0].output_size
                     for _ in range(num_to_generate - generated_count):
                         try:
                             random_ind = create_individual_pytorch(input_s, output_s).to(device)
                             new_population.append(random_ind)
                         except Exception as e:
                             logging.error(f"Failed to create random individual to fill population: {e}")
                else: # İlk popülasyon da boşsa veya hata oluştuysa
                     logging.error("Cannot create random individuals as initial population is unavailable.")


            population = new_population[:pop_size] # Boyutu garantile

            # 4. Checkpoint Alma
            if args.checkpoint_interval > 0 and (gen + 1) % args.checkpoint_interval == 0:
                 try:
                     rnd_state = random.getstate()
                     np_rnd_state = np.random.get_state()
                     torch_rnd_state = torch.get_rng_state().cpu() # CPU state'i kaydet
                     wandb_id = wandb_run.id if wandb_run else None
                     save_checkpoint_pytorch(output_dir, gen + 1, population, rnd_state, np_rnd_state, torch_rnd_state, wandb_id)
                 except Exception as e:
                     logging.error(f"Failed to execute checkpoint saving for generation {gen+1}: {e}", exc_info=True)

            # Bellek temizliği (çok büyük ağlarda işe yarayabilir)
            # import gc; gc.collect()
            # if device.type == 'cuda': torch.cuda.empty_cache()

    finally: # Executor'ı her zaman kapat
        if executor:
            logging.info("Shutting down ProcessPoolExecutor...")
            executor.shutdown(wait=True) # İşlerin bitmesini bekle
            logging.info("Executor shut down.")


    # Evrim Sonu
    if best_model_overall is None and population:
        logging.warning("Evolution finished, but no single best model was tracked. Selecting best from final population.")
        # Son popülasyondan en iyiyi seçmek için fitness'ları tekrar hesapla (veya son skorları kullan?)
        # En güvenlisi tekrar hesaplamak:
        final_population_states = [(ind.get_architecture(), ind.state_dict()) for ind in population]
        final_fitness_scores = [-np.inf] * len(population)
        # Seri hesaplama yapalım (executor kapalı)
        temp_device = torch.device("cpu")
        for i, (arch, state) in enumerate(final_population_states):
             try:
                  model_instance = NeuralNetwork(**arch); model_instance.load_state_dict(state); model_instance.to(temp_device)
                  final_fitness_scores[i] = calculate_fitness_pytorch(model_instance, X_train_np, y_train_np, temp_device, fitness_params)
             except Exception: final_fitness_scores[i] = -np.inf

        final_valid_indices = [i for i, score in enumerate(final_fitness_scores) if np.isfinite(score)]
        if final_valid_indices:
            best_idx_final = max(final_valid_indices, key=lambda i: final_fitness_scores[i])
            best_model_overall = clone_pytorch_model(population[best_idx_final], device)
            best_fitness_overall = final_fitness_scores[best_idx_final]
            logging.info(f"Selected best model from final population: {best_model_overall.model_name} with fitness {best_fitness_overall:.6f}")
        else:
            logging.error("Evolution finished. No valid finite fitness scores in the final population.")
            return None, best_fitness_history, avg_fitness_history
    elif not population:
         logging.error("Evolution finished with an empty population!")
         return None, best_fitness_history, avg_fitness_history
    else: # best_model_overall zaten bulundu
         logging.info(f"Evolution finished. Best fitness achieved: {best_fitness_overall:.6f} by model {best_model_overall.model_name}")

    return best_model_overall, best_fitness_history, avg_fitness_history


# --- Fitness Hesaplama (Seri - Ana Süreç veya Worker=0 için) ---
# Paralel worker'dan farklı olarak modeli doğrudan alır.
def calculate_fitness_pytorch(
    individual: NeuralNetwork,
    X_np: np.ndarray, y_np: np.ndarray, # Veriyi NumPy olarak alır
    device: torch.device,
    fitness_params: Dict
) -> float:
    """ Bir bireyin fitness değerini hesaplar (Seri kullanım için). """
    individual.eval()
    individual.to(device)
    # Veriyi Tensör'e çevir ve cihaza taşı
    try:
        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)
    except Exception as e:
         logging.error(f"Error converting data to tensor or moving to device in calculate_fitness_pytorch: {e}")
         return -np.inf

    complexity_penalty_weight = fitness_params.get('complexity_penalty', 0.0)

    try:
        with torch.no_grad():
            y_pred = individual(X)
            mse_val = torch.mean((y_pred - y)**2).item()

        if not np.isfinite(mse_val):
            logging.warning(f"Non-finite MSE ({mse_val}) for model {individual.model_name} (Serial Calc). Assigning minimal fitness.")
            return -np.inf

        fitness_score = 1.0 / (mse_val + 1e-9)

        if complexity_penalty_weight > 0:
            num_params = individual.get_num_params(trainable_only=True)
            complexity_penalty = complexity_penalty_weight * num_params
            fitness_score -= complexity_penalty

        if not np.isfinite(fitness_score):
             logging.warning(f"Non-finite final fitness ({fitness_score:.4g}) for model {individual.model_name} (Serial Calc). Assigning minimal fitness.")
             return -np.inf

        return float(fitness_score)

    except Exception as e:
        logging.error(f"Error during serial fitness calculation for model {individual.model_name}: {e}", exc_info=True)
        return -np.inf


# --- Grafik Çizimi ---
# (plot_fitness_history fonksiyonu öncekiyle aynı, v4'teki gibi)
def plot_fitness_history(history_best: List[float], history_avg: List[float], output_dir: str, filename: str = "fitness_history_pytorch_v5.png") -> None:
    if not history_best or not history_avg: logging.warning("Fitness history empty, cannot plot."); return
    try:
        plt.figure(figsize=(12, 7))
        gens = np.arange(1, len(history_best) + 1)
        valid_best_indices = [i for i, v in enumerate(history_best) if np.isfinite(v)]
        valid_avg_indices = [i for i, v in enumerate(history_avg) if np.isfinite(v)]
        if valid_best_indices: plt.plot(gens[valid_best_indices], np.array(history_best)[valid_best_indices], label="Best Fitness", marker='o', linestyle='-', linewidth=2)
        if valid_avg_indices: plt.plot(gens[valid_avg_indices], np.array(history_avg)[valid_avg_indices], label="Average Fitness", marker='x', linestyle='--', alpha=0.7)
        plt.xlabel("Generation"); plt.ylabel("Fitness Score"); plt.title("Evolutionary Fitness History (PyTorch v5)"); plt.legend(); plt.grid(True); plt.tight_layout()
        plot_path = os.path.join(output_dir, filename); plt.savefig(plot_path); plt.close()
        logging.info(f"Fitness history plot saved to {plot_path}")
    except Exception as e: logging.error(f"Error plotting fitness history: {e}", exc_info=True)


# --- Değerlendirme (PyTorch v5) ---
# (evaluate_model_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
# Sadece loglamayı güncelleyebiliriz.
def evaluate_model_pytorch(
    model: NeuralNetwork,
    X_test_np: np.ndarray, y_test_np: np.ndarray,
    batch_size: int, device: torch.device
) -> Dict[str, float]:
    """ En iyi modeli test verisi üzerinde PyTorch v5 ile değerlendirir. """
    if model is None: logging.error("Cannot evaluate a None model."); return {"test_mse": np.inf, "avg_kendall_tau": 0.0}
    logging.info(f"Evaluating final model {model.model_name} on test data (PyTorch v5)...")
    model.eval(); model.to(device)
    try:
        test_dataset = TensorDataset(torch.from_numpy(X_test_np).float(), torch.from_numpy(y_test_np).float())
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    except Exception as e:
        logging.error(f"Failed to create PyTorch DataLoader for test data: {e}", exc_info=True)
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}

    all_preds, all_targets = [], []
    total_mse, num_batches = 0.0, 0
    try:
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                total_mse += torch.mean((outputs - targets)**2).item()
                num_batches += 1
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_mse = total_mse / num_batches if num_batches > 0 else np.inf
        logging.info(f"Final Test MSE: {avg_mse:.6f}")
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)
        sample_size = min(500, all_targets_np.shape[0]); taus = []
        if sample_size > 0:
            indices = np.random.choice(all_targets_np.shape[0], sample_size, replace=False)
            for i in indices:
                try:
                    tau, _ = kendalltau(all_targets_np[i], all_preds_np[i])
                    if not np.isnan(tau): taus.append(tau)
                except ValueError: pass
        avg_kendall_tau = np.mean(taus) if taus else 0.0
        logging.info(f"Average Kendall's Tau (on {sample_size} samples): {avg_kendall_tau:.4f}")
        return {"test_mse": float(avg_mse), "avg_kendall_tau": float(avg_kendall_tau)}
    except Exception as e:
        logging.error(f"Error during final model evaluation: {e}", exc_info=True)
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}


# --- Son Eğitim (PyTorch v5) ---
# (train_final_model_pytorch fonksiyonu öncekiyle aynı, v4'teki gibi)
# Sadece loglamayı güncelleyebiliriz.
def train_final_model_pytorch(
    model: NeuralNetwork,
    X_train_np: np.ndarray, y_train_np: np.ndarray,
    epochs: int, batch_size: int, learning_rate: float,
    device: torch.device, output_dir: str,
    wandb_run: Optional[Any] # W&B objesi
) -> Tuple[NeuralNetwork, Dict[str, Any]]:
    """ En iyi evrimleşmiş modeli PyTorch v5 ile eğitir. """
    logging.info(f"--- Starting Final Training of Best Evolved Model ({model.model_name}) ---")
    model.to(device)
    try:
        train_dataset = TensorDataset(torch.from_numpy(X_train_np).float(), torch.from_numpy(y_train_np).float())
        val_split = 0.2; num_train = len(train_dataset); split_idx = int(np.floor(val_split * num_train))
        indices = list(range(num_train)); np.random.shuffle(indices)
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SequentialSampler(val_indices) # Sıralı yapalım val'ı
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=min(4, os.cpu_count() or 1)) # DataLoader workerları
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=min(4, os.cpu_count() or 1))
        logging.info(f"Created DataLoaders. Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    except Exception as e:
        logging.error(f"Failed to create DataLoaders for final training: {e}", exc_info=True)
        return model, {"error": "DataLoader creation failed"}

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=7, verbose=False, min_lr=1e-7) # verbose=False

    early_stopping_patience = 15; best_val_loss = np.inf; epochs_no_improve = 0; best_model_state = None
    training_history = {'train_loss': [], 'val_loss': [], 'lr': []}; epochs_run = 0

    try:
        for epoch in range(epochs):
            epochs_run += 1; model.train(); running_train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, targets)
                loss.backward(); optimizer.step()
                running_train_loss += loss.item()
            avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            training_history['train_loss'].append(avg_train_loss); training_history['lr'].append(optimizer.param_groups[0]['lr'])

            model.eval(); running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    running_val_loss += criterion(model(inputs), targets).item()
            avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else np.inf
            training_history['val_loss'].append(avg_val_loss)
            logging.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # W&B Loglama (final training)
            if wandb_run:
                try:
                    wandb_run.log({
                        "final_train_epoch": epoch + 1,
                        "final_train_loss": avg_train_loss,
                        "final_val_loss": avg_val_loss,
                        "final_learning_rate": optimizer.param_groups[0]['lr']
                    }, step=start_generation + epochs_run) # Toplam adım sayısı
                except Exception as e:
                     logging.warning(f"Failed to log final training metrics to W&B: {e}")


            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict()); logging.debug(f"New best val loss: {best_val_loss:.6f}")
            else: epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs."); break

        if best_model_state: logging.info(f"Restoring model to best validation performance."); model.load_state_dict(best_model_state)
        else: logging.warning("No best model state saved during training.")

        logging.info("Final training complete.")
        training_summary = {"epochs_run": epochs_run, "final_train_loss": avg_train_loss,
                            "best_val_loss": best_val_loss, "final_lr": optimizer.param_groups[0]['lr']}
        return model, training_summary

    except Exception as e:
        logging.error(f"Error during final PyTorch model training: {e}", exc_info=True)
        return model, {"error": str(e)}


# --- Ana İş Akışı (PyTorch v5) ---
def run_pipeline_pytorch_v5(args: argparse.Namespace):
    """ Checkpoint, Adaptif, Paralel PyTorch v5 tabanlı ana iş akışı. """

    wandb_run = None # W&B run objesi
    output_dir = None # Hata durumunda tanımlı olması için

    try: # Ana try bloğu, W&B finish için
        device = setup_device(args.device)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"evorun_pt_v5_{timestamp}_gen{args.generations}_pop{args.pop_size}"
        output_dir = args.resume_from if args.resume_from else os.path.join(args.output_base_dir, run_name)
        resume_run = bool(args.resume_from)
        resumed_wandb_id = None

        if resume_run:
            run_name = os.path.basename(output_dir)
            logging.info(f"Attempting to resume PyTorch v5 run from: {output_dir}")
            if not os.path.isdir(output_dir): logging.error(f"Resume directory not found: {output_dir}. Exiting."); sys.exit(1)
        else:
            try: os.makedirs(output_dir, exist_ok=True)
            except OSError as e: print(f"FATAL: Could not create output dir: {output_dir}. Error: {e}", file=sys.stderr); sys.exit(1)

        setup_logging(output_dir)
        logging.info(f"========== Starting/Resuming EvoNet v5 PyTorch Pipeline: {run_name} ==========")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Using device: {device}")

        # Checkpoint Yükleme
        start_generation = 0; population = []; initial_state_loaded = False; loaded_history_best = []; loaded_history_avg = []
        latest_checkpoint_path = find_latest_checkpoint_pytorch(output_dir) if resume_run else None

        if latest_checkpoint_path:
            loaded_state = load_checkpoint_pytorch(latest_checkpoint_path, device)
            if loaded_state:
                start_generation = loaded_state['generation']
                population = loaded_state['population']
                resumed_wandb_id = loaded_state.get("wandb_run_id") # W&B ID'sini al
                try: # Random state yükleme
                    random.setstate(loaded_state['random_state']); np.random.set_state(loaded_state['numpy_random_state'])
                    torch.set_rng_state(loaded_state['torch_random_state'].cpu())
                    logging.info(f"Random states restored from checkpoint (Generation {start_generation}).")
                except Exception as e: logging.warning(f"Could not fully restore random states: {e}")
                initial_state_loaded = True
                logging.info(f"Resuming from Generation {start_generation + 1} with {len(population)} individuals.")
                if resumed_wandb_id: logging.info(f"Found previous W&B run ID in checkpoint: {resumed_wandb_id}")
            else: logging.error("Failed to load checkpoint. Starting from scratch."); resume_run = False
        elif resume_run: logging.warning(f"Resume requested but no valid v5 checkpoint found. Starting from scratch."); resume_run = False


        # W&B Başlatma (eğer argüman verildiyse ve kütüphane varsa)
        if args.use_wandb and _WANDB_AVAILABLE:
             try:
                 wandb_kwargs = {
                     "project": args.wandb_project,
                     "entity": args.wandb_entity,
                     "name": run_name,
                     "config": vars(args), # Argümanları kaydet
                     "dir": output_dir, # Logları çıktı klasörüne yazdır
                     "resume": "allow", # Devam etmeye izin ver
                     "id": resumed_wandb_id # Eğer varsa önceki ID'yi kullan
                 }
                 # Entity boşsa argümandan çıkar
                 if not wandb_kwargs["entity"]: del wandb_kwargs["entity"]

                 wandb_run = wandb.init(**wandb_kwargs)
                 logging.info(f"Weights & Biases initialized. Run ID: {wandb_run.id if wandb_run else 'N/A'}")
                 # Eğer yeni bir run başladıysa (resume edilmediyse) veya ID değiştiyse W&B ID'sini logla
                 if wandb_run and (not resume_run or wandb_run.id != resumed_wandb_id):
                      logging.info(f"Logging to W&B run: {wandb_run.get_url()}" if wandb_run else "W&B run URL not available.")

             except Exception as e:
                 logging.error(f"Failed to initialize Weights & Biases: {e}", exc_info=True)
                 wandb_run = None # Başarısız olursa devam et ama loglama yapma


        # Config Kaydetme/Loglama (v4'teki gibi)
        config_path = os.path.join(output_dir, "config_pytorch_v5.json")
        args_dict = vars(args)
        if not initial_state_loaded or not os.path.exists(config_path):
             logging.info("--- Configuration ---")
             for k, v in args_dict.items(): logging.info(f"  {k:<25}: {v}")
             logging.info("---------------------")
             try:
                 args_to_save = args_dict.copy(); args_to_save['device'] = str(device)
                 with open(config_path, 'w') as f: json.dump(args_to_save, f, indent=4, sort_keys=True)
                 logging.info(f"Configuration saved to {config_path}")
             except Exception as e: logging.error(f"Failed to save configuration: {e}", exc_info=True)
        else: # Devam ediliyorsa logla
            try:
                 with open(config_path, 'r') as f: loaded_args_dict = json.load(f)
                 logging.info("--- Loaded Configuration (from resumed run) ---")
                 for k, v in loaded_args_dict.items(): logging.info(f"  {k:<25}: {v}")
                 logging.info("-----------------------------------------------")
            except Exception as e: logging.warning(f"Could not reload config.json: {e}")


        # Random Tohum Ayarlama (sadece sıfırdan başlarken)
        if not initial_state_loaded:
            try:
                seed = args.seed
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
                logging.info(f"Using random seed: {seed}")
            except Exception as e: logging.warning(f"Could not set all random seeds: {e}")


        # Veri Üretimi (her zaman)
        try:
            logging.info("Generating/Reloading data...")
            X_train_np, y_train_np = generate_data(args.train_samples, args.seq_length)
            X_test_np, y_test_np = generate_data(args.test_samples, args.seq_length)
            input_shape = X_train_np.shape[1]
            output_shape = y_train_np.shape[1]
        except Exception: logging.critical("Failed to generate/reload data. Exiting."); sys.exit(1)


        # Popülasyon Başlatma (sadece sıfırdan başlarken)
        if not initial_state_loaded:
            logging.info(f"--- Initializing Population (Size: {args.pop_size}) ---")
            try:
                population = [create_individual_pytorch(input_shape, output_shape).to(device) for _ in range(args.pop_size)]
                logging.info("Population initialized successfully.")
            except Exception: logging.critical("Failed to initialize population. Exiting."); sys.exit(1)


        # Evrim Süreci
        logging.info(f"--- Starting/Resuming PyTorch v5 Evolution ({args.generations} Total Generations) ---")
        best_model_evolved: Optional[NeuralNetwork] = None
        best_fitness_hist = loaded_history_best
        avg_fitness_hist = loaded_history_avg

        if start_generation >= args.generations:
            logging.warning(f"Loaded checkpoint gen ({start_generation}) >= total gens ({args.generations}). Skipping evolution.")
            # Checkpoint'ten en iyiyi al (v4'teki gibi TODO: daha iyi yöntem)
            if population:
                # Son popülasyondan en iyiyi seç (fitness hesaplayarak)
                 try:
                     logging.info("Selecting best model from loaded population as evolution is skipped...")
                     temp_device = torch.device("cpu")
                     fitness_scores_loaded = [calculate_fitness_pytorch(ind, X_train_np, y_train_np, temp_device, {'complexity_penalty': args.complexity_penalty}) for ind in population]
                     valid_scores_loaded = [(s, i) for i, s in enumerate(fitness_scores_loaded) if np.isfinite(s)]
                     if valid_scores_loaded:
                         best_idx_loaded = max(valid_scores_loaded, key=lambda item: item[0])[1]
                         best_model_evolved = clone_pytorch_model(population[best_idx_loaded], device)
                         logging.info(f"Using model {best_model_evolved.model_name} from loaded population.")
                     else: logging.warning("Could not determine best model from loaded population."); best_model_evolved = None
                 except Exception as e: logging.error(f"Error selecting best model from loaded population: {e}"); best_model_evolved = None
            else: best_model_evolved = None
        else:
            try:
                best_model_evolved, gen_best_hist, gen_avg_hist = evolve_population_pytorch_v5(
                    population, X_train_np, y_train_np, start_generation, args.generations,
                    args.crossover_rate, args.mutation_rate, args.weight_mut_rate,
                    args, # Tüm argümanları geçir
                    output_dir, device, wandb_run
                )
                best_fitness_hist.extend(gen_best_hist)
                avg_fitness_hist.extend(gen_avg_hist)
            except Exception as e:
                logging.critical(f"Fatal error during PyTorch v5 evolution process: {e}", exc_info=True)
                raise # Hatayı yukarı fırlat

        logging.info("--- PyTorch v5 Evolution Complete ---")

        # Fitness Geçmişi Kaydet/Çizdir (v4'teki gibi)
        if best_fitness_hist or avg_fitness_hist:
            plot_fitness_history(best_fitness_hist, avg_fitness_hist, output_dir)
            history_path = os.path.join(output_dir, "fitness_history_pytorch_v5.csv")
            try:
                history_data = np.array([np.arange(1, len(best_fitness_hist) + 1), best_fitness_hist, avg_fitness_hist]).T
                np.savetxt(history_path, history_data, delimiter=',', header='Generation,BestFitness,AvgFitness', comments='', fmt=['%d', '%.8f', '%.8f'])
                logging.info(f"Full fitness history saved to {history_path}")
                # W&B'ye tablo olarak logla (opsiyonel)
                if wandb_run:
                    try:
                         table = wandb.Table(data=history_data, columns=["Generation", "BestFitness", "AvgFitness"])
                         wandb_run.log({"fitness_history_table": table})
                    except Exception as e: logging.warning(f"Failed to log fitness history table to W&B: {e}")

            except Exception as e: logging.error(f"Could not save fitness history data: {e}")
        else: logging.warning("Fitness history empty, skipping saving/plotting.")

        # En İyi Modeli Eğit/Değerlendir/Kaydet
        final_model_path = None; training_summary = {}; final_metrics = {"test_mse": np.inf, "avg_kendall_tau": 0.0}; best_model_architecture = {}
        if best_model_evolved is None:
            logging.error("Evolution did not yield a best model. Skipping final training and evaluation.")
        else:
            best_model_architecture = best_model_evolved.get_architecture()
            logging.info(f"Best evolved model architecture: {best_model_architecture}")
            try:
                num_params = best_model_evolved.get_num_params(); logging.info(f"Best Evolved Model ({best_model_evolved.model_name}) - Params: {num_params}")
                if wandb_run: wandb_run.summary["best_evolved_params"] = num_params # W&B özete ekle
            except Exception as e: logging.warning(f"Could not log model summary details: {e}")

            # Son Eğitim
            try:
                 model_to_train = clone_pytorch_model(best_model_evolved, device)
                 final_model, training_summary = train_final_model_pytorch(
                     model_to_train, X_train_np, y_train_np,
                     args.epochs_final_train, args.batch_size, args.learning_rate,
                     device, output_dir, wandb_run
                 )
            except Exception as e: logging.error(f"Error during final training: {e}", exc_info=True); final_model = None; training_summary = {"error": str(e)}

            # Değerlendirme ve Kaydetme
            if final_model:
                 final_metrics = evaluate_model_pytorch(final_model, X_test_np, y_test_np, args.batch_size, device)
                 if wandb_run: wandb_run.summary.update(final_metrics) # W&B özete ekle

                 final_model_path = os.path.join(output_dir, "best_evolved_model_trained_pytorch_v5.pt")
                 try:
                     torch.save({'architecture': final_model.get_architecture(), 'model_state_dict': final_model.state_dict(),
                                 'training_summary': training_summary, 'evaluation_metrics': final_metrics}, final_model_path)
                     logging.info(f"Final trained model state and architecture saved to {final_model_path}")
                     # W&B'ye artifact olarak kaydet (opsiyonel)
                     if wandb_run:
                          try:
                               artifact = wandb.Artifact(f'final_model_{run_name}', type='model')
                               artifact.add_file(final_model_path)
                               wandb_run.log_artifact(artifact)
                               logging.info(f"Saved final model as W&B artifact.")
                          except Exception as e: logging.warning(f"Failed to save model as W&B artifact: {e}")
                 except Exception as e: logging.error(f"Failed to save final trained model: {e}", exc_info=True); final_model_path = None
            else: logging.error("Final model training failed. Skipping evaluation and saving.")

        # Sonuçları Kaydet
        logging.info("--- Saving Final Results (v5) ---")
        final_results = {
            "run_info": {"run_name": run_name, "timestamp": timestamp, "output_directory": output_dir, "framework": "PyTorch",
                         "version": "v5", "device_used": str(device), "resumed_run": resume_run, "last_checkpoint": latest_checkpoint_path,
                         "wandb_url": wandb_run.get_url() if wandb_run else None},
            "config": args_dict,
            "evolution_summary": {
                "start_generation": start_generation, "end_generation": start_generation + len(best_fitness_hist),
                "generations_run_this_session": len(best_fitness_hist) - len(loaded_history_best),
                "best_fitness_overall": max(best_fitness_hist) if best_fitness_hist and any(np.isfinite(f) for f in best_fitness_hist) else None,
                "best_fitness_final_gen": best_fitness_hist[-1] if best_fitness_hist and np.isfinite(best_fitness_hist[-1]) else None,
                "avg_fitness_final_gen": avg_fitness_hist[-1] if avg_fitness_hist and np.isfinite(avg_fitness_hist[-1]) else None,
                "best_model_architecture": best_model_architecture,
                "best_model_params": best_model_evolved.get_num_params() if best_model_evolved else None
            },
            "final_training_summary": training_summary,
            "final_evaluation_on_test": final_metrics,
            "saved_trained_model_path": final_model_path
        }
        results_path = os.path.join(output_dir, "final_results_pytorch_v5.json")
        try:
            def convert_types(obj): # JSON için tür dönüştürücü
                if isinstance(obj, (np.integer, np.int_)): return int(obj)
                elif isinstance(obj, (np.floating, np.float_)): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                elif isinstance(obj, torch.Tensor): return obj.tolist()
                elif isinstance(obj, torch.device): return str(obj)
                elif isinstance(obj, type): return obj.__name__
                elif isinstance(obj, argparse.Namespace): return vars(obj) # Argümanları dict yap
                return obj
            with open(results_path, 'w') as f: json.dump(final_results, f, indent=4, default=convert_types, sort_keys=True)
            logging.info(f"Final results summary saved to {results_path}")
        except Exception as e: logging.error(f"Failed to save final results JSON: {e}", exc_info=True)

    except (Exception, KeyboardInterrupt) as e:
         # Hata veya kesinti durumunda loglama ve W&B bitirme
         if isinstance(e, KeyboardInterrupt):
              logging.warning("KeyboardInterrupt detected. Exiting.")
         else:
              logging.critical("Unhandled exception in pipeline:", exc_info=True)
         # W&B run'ı "crashed" veya "failed" olarak işaretle
         if wandb_run:
              exit_code = 1 if not isinstance(e, KeyboardInterrupt) else 130
              try:
                   wandb.finish(exit_code=exit_code, quiet=True)
                   logging.info(f"W&B run marked as {'failed' if exit_code==1 else 'killed'}.")
              except Exception as wb_e:
                   logging.error(f"Error finishing W&B run: {wb_e}")
         # Hatayı tekrar fırlat veya çık
         if isinstance(e, KeyboardInterrupt): sys.exit(130)
         else: sys.exit(1)

    finally:
        # W&B run'ı normal şekilde bitir (eğer hata olmadıysa)
        if wandb_run and not sys.exc_info()[0]: # Sadece hata yoksa bitir
             try:
                 wandb.finish()
                 logging.info("W&B run finished successfully.")
             except Exception as e:
                 logging.error(f"Error finishing W&B run: {e}")

        logging.info(f"========== PyTorch v5 Pipeline Run {run_name} Finished ==========")


# --- Argüman Ayrıştırıcı (v5) ---
def parse_arguments_v5() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoNet v5: Adaptive & Parallel Neuroevolution with PyTorch")

    # --- Dizinler ve Kontrol ---
    parser.add_argument('--output_base_dir', type=str, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument('--resume_from', type=str, default=None, help='Path to previous run dir to resume.')
    parser.add_argument('--checkpoint_interval', type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help='Checkpoint frequency (gens). 0=disable.')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: random).')

    # --- Veri ---
    parser.add_argument('--seq_length', type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument('--train_samples', type=int, default=5000)
    parser.add_argument('--test_samples', type=int, default=1000)

    # --- Evrim Parametreleri ---
    evo_group = parser.add_argument_group('Evolution Parameters')
    evo_group.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE)
    evo_group.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS)
    evo_group.add_argument('--crossover_rate', type=float, default=DEFAULT_CROSSOVER_RATE)
    evo_group.add_argument('--mutation_rate', type=float, default=DEFAULT_MUTATION_RATE, help='Prob. of mutation if crossover is not applied.')
    evo_group.add_argument('--weight_mut_rate', type=float, default=DEFAULT_WEIGHT_MUT_RATE, help='Prob. for each weight to mutate if mutation occurs.')
    evo_group.add_argument('--tournament_size', type=int, default=DEFAULT_TOURNAMENT_SIZE)
    evo_group.add_argument('--elitism_count', type=int, default=DEFAULT_ELITISM_COUNT)
    evo_group.add_argument('--complexity_penalty', type=float, default=DEFAULT_COMPLEXITY_PENALTY, help='Penalty weight per parameter in fitness.')

    # --- Adaptif Mutasyon ---
    adapt_group = parser.add_argument_group('Adaptive Mutation')
    adapt_group.add_argument('--adapt_mutation', action=argparse.BooleanOptionalAction, default=DEFAULT_ADAPT_MUTATION, help='Enable adaptive mutation strength.')
    adapt_group.add_argument('--mutation_strength', type=float, default=DEFAULT_MUTATION_STRENGTH, help='Initial mutation strength (std dev).')
    adapt_group.add_argument('--stagnation_limit', type=int, default=DEFAULT_STAGNATION_LIMIT, help='Generations without improvement to trigger adaptation.')
    adapt_group.add_argument('--mut_strength_decay', type=float, default=DEFAULT_MUT_STRENGTH_DECAY, help='Factor to decrease strength on improvement.')
    adapt_group.add_argument('--mut_strength_increase', type=float, default=DEFAULT_MUT_STRENGTH_INCREASE, help='Factor to increase strength on stagnation.')
    adapt_group.add_argument('--min_mut_strength', type=float, default=DEFAULT_MIN_MUT_STRENGTH)
    adapt_group.add_argument('--max_mut_strength', type=float, default=DEFAULT_MAX_MUT_STRENGTH)

    # --- Paralellik ---
    parallel_group = parser.add_argument_group('Parallelism')
    parallel_group.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='Number of CPU workers for parallel fitness evaluation (0=disable/serial).')

    # --- Eğitim ve Değerlendirme ---
    train_group = parser.add_argument_group('Final Training & Evaluation')
    train_group.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    train_group.add_argument('--epochs_final_train', type=int, default=DEFAULT_EPOCHS_FINAL_TRAIN)
    train_group.add_argument('--learning_rate', type=float, default=0.001, help='LR for final training.')

    # --- Deney Takibi (W&B) ---
    wandb_group = parser.add_argument_group('Experiment Tracking (Weights & Biases)')
    wandb_group.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=False, help='Enable W&B logging.')
    wandb_group.add_argument('--wandb_project', type=str, default="EvoNet-v5", help='W&B project name.')
    wandb_group.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team). Uses default if None.') # Genellikle kullanıcı adı veya takım

    args = parser.parse_args()
    if args.seed is None: args.seed = random.randint(0, 2**32 - 1); print(f"Generated random seed: {args.seed}")
    if args.num_workers < 0: print(f"Warning: num_workers ({args.num_workers}) cannot be negative. Setting to 0."); args.num_workers = 0
    # Diğer v4 kontrolleri (elitism, tournament size) burada da yapılabilir.

    return args

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # Önemli Not: concurrent.futures (özellikle ProcessPoolExecutor) ve
    # multiprocessing'in düzgün çalışması için ana kod bloğunun
    # `if __name__ == "__main__":` içinde olması genellikle gereklidir.
    cli_args = parse_arguments_v5()
    run_pipeline_pytorch_v5(cli_args)