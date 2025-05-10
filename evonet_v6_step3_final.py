# ==============================================================================
# EvoNet Optimizer - v6 (Adım 3: Final - Çaprazlama Eklendi)
# Açıklama: v6 Adım 2 (Türleşme) üzerine inşa edilmiştir. NEAT tarzı
#           çaprazlama mekanizması eklenmiştir. Bu sürüm, Genom yapısı,
#           yapısal mutasyonlar, türleşme ve çaprazlamayı içerir.
# ==============================================================================

import os
# os.environ["WANDB_SILENT"] = "true"
import sys
import argparse
import random
import logging
from datetime import datetime
import json
import copy
import time
from typing import List, Tuple, Dict, Any, Optional, Union, Set
import concurrent.futures
from enum import Enum, auto
import math

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

# --- Sabitler ve Varsayılan Değerler (v6 Final için güncellemeler) ---
# ... (Adım 2'deki sabitler: POP_SIZE, GENERATIONS, mutasyon oranları, türleşme parametreleri vb.) ...
DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "evonet_runs_v6_pytorch")
DEFAULT_CROSSOVER_RATE = 0.75 # Çaprazlama uygulama olasılığı (yüksek tutulabilir)
# DEFAULT_INTERSPECIES_MATE_RATE = 0.01 # Farklı türler arası çiftleşme olasılığı (şimdilik eklenmedi)
DEFAULT_MATE_BY_AVERAGING = False # Eşleşen genlerde ağırlıkları ortala (False = rastgele seç)

# --- Loglama, Cihaz Ayarları, Veri Üretimi (Adım 1/2'den aynı) ---
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    # ... (Adım 2 ile aynı, log dosya adı v6 olabilir) ...
    log_filename = os.path.join(log_dir, 'evolution_run_pytorch_v6_final.log')
    # ... (kalanı aynı) ...
def setup_device(requested_device: str) -> torch.device: pass # ... (Önceki kodu buraya yapıştırın) ...
def generate_data(num_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]: pass # ... (Önceki kodu buraya yapıştırın) ...


# --- v6 Genom Yapısı (Adım 2'den aynı, crossover metodu eklendi) ---
class NodeType(Enum): INPUT = auto(); OUTPUT = auto(); HIDDEN = auto()
class NodeGene: # ... (Adım 1/2 ile aynı) ...
    pass
class ConnectionGene: # ... (Adım 1/2 ile aynı) ...
    pass
class InnovationTracker: # ... (Adım 1/2 ile aynı) ...
    pass

class Genome: # Adım 2'den gelen Genome sınıfı + crossover
    _genome_counter = 0
    def __init__(self, genome_id: Optional[int] = None):
        self.id = genome_id if genome_id is not None else Genome._genome_counter
        Genome._genome_counter += 1
        self.node_genes: Dict[int, NodeGene] = {}
        self.connection_genes: Dict[int, ConnectionGene] = {}
        self.fitness: Optional[float] = None
        self.adjusted_fitness: Optional[float] = None
        self.species_id: Optional[int] = None

    # --- add_node_gene, add_connection_gene (Adım 1/2'den aynı) ---
    # ...

    # --- mutate ve alt mutasyon fonksiyonları (Adım 1/2'den aynı) ---
    # ...

    def crossover(self, other_parent: 'Genome', child_id: Optional[int] = None,
                  mate_by_avg: bool = False) -> 'Genome':
        """ İki genomu NEAT tarzı çaprazlayarak bir çocuk genomu oluşturur. """
        child = Genome(child_id) # Yeni çocuk için ID ata

        # Hangi ebeveynin daha fit olduğunu belirle (orijinal fitness'a göre)
        fitter_parent = self
        less_fit_parent = other_parent
        # None kontrolü önemli
        fit1 = self.fitness if self.fitness is not None and np.isfinite(self.fitness) else -np.inf
        fit2 = other_parent.fitness if other_parent.fitness is not None and np.isfinite(other_parent.fitness) else -np.inf

        if fit1 < fit2:
             fitter_parent = other_parent
             less_fit_parent = self
        elif fit1 == fit2:
             # Fitness eşitse, daha küçük olanı (daha az gene sahip) fitter kabul edelim mi?
             # Veya rastgele seçelim? Şimdilik rastgele seçelim.
             if random.random() < 0.5:
                  fitter_parent, less_fit_parent = less_fit_parent, fitter_parent

        # --- Düğüm Genleri ---
        # Basit yaklaşım: Her iki ebeveyndeki tüm düğümleri çocuğa aktar.
        # Daha iyisi: Sadece kullanılan bağlantıların gerektirdiği düğümleri aktarmak.
        # Şimdilik tüm düğümleri fitter'dan alalım (daha basit).
        # VEYA: Union alalım, bu daha güvenli.
        child_node_ids = set(self.node_genes.keys()) | set(other_parent.node_genes.keys())
        for node_id in child_node_ids:
             # Eğer düğüm her iki ebeveynde de varsa, fitter'dan alalım (veya rastgele?)
             # Şimdilik varsa fitter'dan, yoksa diğerinden alalım.
             node1 = self.node_genes.get(node_id)
             node2 = other_parent.node_genes.get(node_id)
             if node1 and node2:
                  child.add_node_gene(fitter_parent.node_genes[node_id].copy())
             elif node1:
                  child.add_node_gene(node1.copy())
             elif node2:
                  child.add_node_gene(node2.copy())


        # --- Bağlantı Genleri ---
        innovs1 = set(self.connection_genes.keys())
        innovs2 = set(other_parent.connection_genes.keys())
        max_parent_innov = max(max(innovs1) if innovs1 else 0, max(innovs2) if innovs2 else 0)

        for innov in range(max_parent_innov + 1):
            conn1 = self.connection_genes.get(innov)
            conn2 = other_parent.connection_genes.get(innov)
            child_conn: Optional[ConnectionGene] = None

            if conn1 is not None and conn2 is not None: # Eşleşen Gen
                # Rastgele seç veya ortalama al
                if mate_by_avg:
                    chosen_conn = conn1.copy() # Birinden kopyala
                    chosen_conn.weight = (conn1.weight + conn2.weight) / 2.0 # Ortalamasını al
                    child_conn = chosen_conn
                else: # Rastgele seç
                    chosen_conn = random.choice([conn1, conn2])
                    child_conn = chosen_conn.copy()

                # Eğer bir ebeveynde devre dışıysa, çocukta da devre dışı olma ihtimali
                # (NEAT'teki kural: %75 ihtimalle devre dışı kalır)
                if not conn1.enabled or not conn2.enabled:
                    if random.random() < 0.75:
                         if child_conn: child_conn.enabled = False # Kopyalandıktan sonra değiştir
            elif conn1 is not None and conn2 is None: # Ayrık/Fazla Gen (sadece self'de)
                if fitter_parent is self: # Eğer self fitter ise miras al
                    child_conn = conn1.copy()
            elif conn1 is None and conn2 is not None: # Ayrık/Fazla Gen (sadece other'da)
                if fitter_parent is other_parent: # Eğer other fitter ise miras al
                     child_conn = conn2.copy()

            # Seçilen bağlantı genini çocuğa ekle (ve gerekli düğümlerin var olduğundan emin ol - yukarıda hallettik)
            if child_conn is not None:
                 # Güvenlik: Bağlantının işaret ettiği düğümler çocukta var mı? (Union aldığımız için olmalı)
                 if child_conn.in_node_id in child.node_genes and child_conn.out_node_id in child.node_genes:
                      child.add_connection_gene(child_conn)
                 # else: logging.warning(f"Crossover skipped conn {child_conn.innovation_number} due to missing nodes in child.")


        return child


    # ... (get_phenotype_model, copy, __repr__, to_dict, from_dict - Adım 1/2'den aynı) ...
    pass

# --- v6 Fenotip Modeli (Adım 1/2'den aynı) ---
# ... (FeedForwardNetwork sınıfı ve _sigmoid vb. fonksiyonlar) ...
pass

# --- v6 Uyumluluk Mesafesi (Adım 2'den aynı) ---
def calculate_compatibility_distance(genome1: Genome, genome2: Genome, c1: float, c2: float, c3: float) -> float: pass # ... (Adım 2'den aynı) ...

# --- v6 Tür (Species) Sınıfı (Adım 2'den aynı) ---
class Species: pass # ... (Adım 2'den aynı) ...


# --- v6 Evrim Döngüsü (Türleşme + Çaprazlama ile) ---
def evolve_population_pytorch_v6_final( # Fonksiyon adı güncellendi
    population: List[Genome],
    X_train_np: np.ndarray, y_train_np: np.ndarray,
    start_generation: int, total_generations: int,
    args: argparse.Namespace,
    output_dir: str, device: torch.device,
    innovation_tracker: InnovationTracker,
    species_list: List[Species], # Tür listesi artık dışarıdan yönetiliyor
    wandb_run: Optional[Any]
) -> Tuple[Optional[Genome], List[float], List[float], List[Species]]: # Güncellenmiş tür listesini döndürür
    """ PyTorch v6 Final: Genom + Türleşme + Çaprazlama ile evrim. """

    best_fitness_history = []
    avg_fitness_history = []
    best_genome_overall: Optional[Genome] = None
    best_fitness_overall = -np.inf

    current_mutation_strength = args.mutation_strength
    stagnation_counter_global = 0 # TODO: Global tıkanma takibi eklenebilir

    pop_size = len(population)
    fitness_params = {'complexity_penalty': args.complexity_penalty}

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) if args.num_workers > 0 else None
    # ... (Executor loglama) ...

    try:
        for gen in range(start_generation, total_generations):
            generation_start_time = time.time()

            # --- 1. Fitness Değerlendirme (Adım 2'deki gibi) ---
            # ... (Paralel/Seri fitness hesaplama, genomların .fitness'ı güncellenir) ...
            # >>> Buraya Adım 2 / Adım 1'deki fitness hesaplama bloğunu ekleyin <<<

            # --- 2. İstatistikler ve En İyiyi Takip (Adım 2'deki gibi - Orijinal fitness) ---
            # ... (best_fitness_overall, avg_fitness, best_genome_overall takibi) ...
            # >>> Buraya Adım 2 / Adım 1'deki istatistik/takip bloğunu ekleyin <<<

            # --- 3. Türleştirme (Adım 2'deki gibi) ---
            logging.debug(f"Generation {gen+1}: Starting speciation...")
            # Tür üyelerini temizle
            for s in species_list: s.reset()
            newly_created_species_count = 0; assigned_genomes = 0
            for genome in population:
                 if genome.fitness is None or not np.isfinite(genome.fitness): continue
                 found_species = False
                 for s in species_list:
                      dist = calculate_compatibility_distance(genome, s.representative, args.c1_excess, args.c2_disjoint, args.c3_weight)
                      if dist < args.compatibility_threshold: s.add_member(genome); found_species = True; assigned_genomes += 1; break
                 if not found_species: new_species = Species(genome); species_list.append(new_species); newly_created_species_count += 1; assigned_genomes += 1
            # ... (Adım 2'deki boş türleri temizleme, hata kontrolü) ...

            # --- 4. Fitness Paylaşımı ve İstatistik Güncelleme (Adım 2'deki gibi) ---
            total_adjusted_fitness = 0.0
            for s in species_list: s.adjust_fitnesses(); s.update_stats(); # ... (total_adjusted_fitness hesaplama) ...

            # --- 5. Durağan Türleri Eleme ve Yavru Sayısını Hesaplama (Adım 2'deki gibi) ---
            pop_to_fill = pop_size - args.elitism_count
            offspring_counts_float = []
            num_remaining_species = 0
            for s in species_list:
                 is_stagnant = s.generations_since_improvement > args.species_stagnation_limit
                 should_protect = len(species_list) <= 2 or (best_genome_overall and best_genome_overall.species_id == s.id)
                 if is_stagnant and not should_protect: logging.info(f"Species {s.id} removed due to stagnation."); continue # Elenen türü atla
                 num_remaining_species += 1
                 if total_adjusted_fitness > 0:
                      offspring_f = s.calculate_offspring_count(total_adjusted_fitness, pop_size, args.elitism_count)
                      offspring_counts_float.append((offspring_f, s))
                 else: offspring_counts_float.append((0.0, s)) # Hata durumunda 0

            # Eğer türler elendiyse species_list'i güncelle
            species_list = [item[1] for item in offspring_counts_float]
            if not species_list: raise RuntimeError("All species died out after stagnation removal!") # Hata ver

            # Yavru sayılarını tam sayıya çevir
            # ... (Adım 2'deki offspring_counts_int hesaplama ve yuvarlama yönetimi) ...
            # >>> Buraya Adım 2'deki offspring tamsayı hesaplama bloğunu ekleyin <<<
            final_offspring_map = {} # {s.id: count}
            # ... Hesaplama sonrası final_offspring_map doldurulur ...
            for s in species_list: s.offspring_to_produce = final_offspring_map.get(s.id, 0)


            # --- 6. Yeni Popülasyon Oluşturma (Çaprazlama + Mutasyon) ---
            new_population = []

            # 6a. Elitizm (Genel en iyiler - Orijinal fitness'a göre)
            if args.elitism_count > 0:
                # ... (Adım 2'deki elitizm kodu) ...
                pass # Önceki kodu buraya yapıştırın

            # 6b. Yavruları Üretme
            generated_offspring = 0
            for s in species_list:
                if not s.members or s.offspring_to_produce <= 0: continue

                # Yavru sayısı kadar döngü
                for i in range(s.offspring_to_produce):
                     if generated_offspring >= pop_to_fill: break
                     try:
                          child_genome: Optional[Genome] = None
                          # Çaprazlama olasılığı (tür içinde en az 2 üye varsa)
                          if len(s.members) > 1 and random.random() < args.crossover_rate:
                               parent1 = s.select_parent()
                               parent2 = s.select_parent()
                               # Ebeveynlerin aynı olmamasını sağlamak iyi olabilir
                               attempts = 0
                               while parent1 is parent2 and attempts < 5 and len(s.members) > 1:
                                    parent2 = s.select_parent()
                                    attempts += 1

                               if parent1 is not parent2:
                                    logging.debug(f"Performing crossover in species {s.id} between {parent1.id} and {parent2.id}")
                                    # Fitter olanı belirleyip çaprazla
                                    fit1 = parent1.fitness if parent1.fitness is not None else -np.inf
                                    fit2 = parent2.fitness if parent2.fitness is not None else -np.inf
                                    if fit1 >= fit2: child_genome = parent1.crossover(parent2, mate_by_avg=args.mate_by_avg)
                                    else: child_genome = parent2.crossover(parent1, mate_by_avg=args.mate_by_avg)
                               else: # Aynı ebeveyn seçildiyse çaprazlama yapma
                                    child_genome = None # Aseksüel üremeye geç
                          # Aseksüel Üreme (Çaprazlama olmadıysa veya tek üye varsa)
                          if child_genome is None:
                               parent = s.select_parent()
                               child_genome = parent.copy() # Sadece kopyala, mutasyon aşağıda

                          # Yavruyu Mutasyona Uğrat
                          if child_genome is not None:
                               args.current_mutation_strength = current_mutation_strength # Adaptif gücü ayarla
                               child_genome.mutate(innovation_tracker, args)
                               del args.current_mutation_strength # Geri sil
                               new_population.append(child_genome)
                               generated_offspring += 1
                          else:
                               logging.warning(f"Failed to produce offspring for species {s.id} (child_genome is None).")

                     except Exception as e:
                          logging.error(f"Error producing offspring for species {s.id}: {e}", exc_info=True)
                if generated_offspring >= pop_to_fill: break


            # Eksik kalırsa doldurma (Adım 2'deki gibi)
            # ... (Adım 2'deki popülasyonu tamamlama kodu) ...

            population = new_population[:pop_size]

            # --- 7. Checkpoint Alma (v6 - Tür durumu ile) ---
            if args.checkpoint_interval > 0 and (gen + 1) % args.checkpoint_interval == 0:
                 try:
                      # ... (Adım 2'deki checkpoint alma kodu - save_checkpoint_pytorch_v6_speciation kullanılır) ...
                      pass # Önceki kodu buraya yapıştırın
                 except Exception as e: logging.error(f"Failed checkpoint saving: {e}", exc_info=True)


            # --- Adaptif Mutasyon Gücü Güncelleme (Adım 2'deki gibi) ---
            # ... (new_best_found kontrolü ve current_mutation_strength güncelleme) ...
            # >>> Buraya Adım 2'deki adaptif mutasyon bloğunu ekleyin <<<

            # --- Jenerasyon Sonu Loglama/W&B (Adım 2'deki gibi) ---
            # ... (W&B loglama dahil) ...
            # >>> Buraya Adım 2'deki jenerasyon sonu loglama bloğunu ekleyin <<<


    finally: # Executor'ı kapat
        if executor: executor.shutdown(wait=True)


    # Evrim Sonu (En iyi genomu seç - Adım 2'deki gibi)
    # ... (Adım 2'deki final seçim kodu) ...
    pass # Önceki kodu buraya yapıştırın

    # Güncellenmiş tür listesini de döndür (checkpoint için önemli olabilir)
    return best_genome_overall, best_fitness_history, avg_fitness_history, species_list


# --- v6 Final: Checkpointing (Tür Durumu ile - Adım 2 ile aynı) ---
def save_checkpoint_pytorch_v6_speciation(*args, **kwargs): pass # Adım 2'den al
# --- v6 Final: Checkpoint Yükleme (Tür Durumu ile - Adım 2 ile aynı) ---
def load_checkpoint_pytorch_v6_speciation(*args, **kwargs): pass # Adım 2'den al

# --- v6 Başlangıç Genomu Oluşturma (Adım 1/2 ile aynı) ---
def create_initial_genome(*args, **kwargs): pass # Adım 1/2'den al
# --- Grafik/Değerlendirme (v6 Adım 2 ile aynı) ---
def plot_fitness_history(*args, **kwargs): pass
def evaluate_model_pytorch_v6(*args, **kwargs): pass


# --- Ana İş Akışı (PyTorch v6 Final) ---
def run_pipeline_pytorch_v6_final(args: argparse.Namespace):
     wandb_run = None; output_dir = None; species_list: List[Species] = []
     try:
        # --- Başlangıç Ayarları (Adım 2 / v5'teki gibi) ---
        # device, timestamp, run_name, output_dir, resume_run, logging setup
        # ...

        # --- Checkpoint Yükleme (v6 Speciation) ---
        start_generation = 0; population: List[Genome] = []; initial_state_loaded = False; tracker_state = None; resumed_wandb_id = None
        latest_checkpoint_path = find_latest_checkpoint_pytorch(output_dir) if resume_run else None # v6 klasörü
        if latest_checkpoint_path:
             loaded_state = load_checkpoint_pytorch_v6_speciation(latest_checkpoint_path)
             if loaded_state:
                 start_generation = loaded_state['generation']
                 population = loaded_state['population']
                 species_list = loaded_state.get('species_list', []) # Tür listesini yükle
                 tracker_state = loaded_state.get("innovation_tracker_state")
                 resumed_wandb_id = loaded_state.get("wandb_run_id")
                 # ... (random state yükleme) ...
                 initial_state_loaded = True
                 logging.info(f"Resuming from Gen {start_generation + 1} with {len(population)} genomes and {len(species_list)} species structures.")
             # ... (yükleme başarısızlık durumu) ...
        # ... (resume=True ama checkpoint yok durumu) ...

        # --- Innovation Tracker Başlatma/Yükleme (Adım 2'deki gibi) ---
        # ...

        # --- W&B Başlatma (Adım 2 / v5'teki gibi) ---
        # ...

        # --- Config Kaydetme/Loglama (Adım 2 / v5'teki gibi) ---
        # ...

        # --- Random Tohum Ayarlama (Adım 2 / v5'teki gibi) ---
        # ...

        # --- Veri Üretimi (Adım 2 / v5'teki gibi) ---
        # ... (input_size, output_size alınır) ...

        # --- Popülasyon Başlatma (Adım 2 / v5'teki gibi) ---
        if not initial_state_loaded:
            # ... (create_initial_genome ile popülasyon oluşturma) ...
            pass

        # --- Evrim Süreci (v6 Final - Çaprazlama ile) ---
        logging.info(f"--- Starting/Resuming PyTorch v6 Final Evolution (Speciation + Crossover) ---")
        best_genome_evolved: Optional[Genome] = None
        best_fitness_hist = [] # TODO: yükle
        avg_fitness_hist = []  # TODO: yükle

        if start_generation >= args.generations:
            # ... (Evrim atlama ve en iyi genomu seçme) ...
            pass
        else:
             try:
                 # FİNAL evrim fonksiyonunu çağır
                 best_genome_evolved, gen_best_hist, gen_avg_hist, final_species_list = evolve_population_pytorch_v6_final(
                     population, X_train_np, y_train_np, start_generation, args.generations,
                     args, output_dir, device, innovation_tracker,
                     species_list, # Başlangıç tür listesi
                     wandb_run
                 )
                 best_fitness_hist.extend(gen_best_hist)
                 avg_fitness_hist.extend(gen_avg_hist)
                 species_list = final_species_list # Son tür durumunu al
             except Exception as e:
                 logging.critical(f"Fatal error during PyTorch v6 final evolution: {e}", exc_info=True)
                 raise

        logging.info("--- PyTorch v6 Final Evolution Complete ---")

        # --- Fitness Geçmişi Kaydet/Çizdir (Adım 2 / v5'teki gibi) ---
        # ...

        # --- En İyi Genomu Değerlendir/Kaydet (Adım 2 / v5'teki gibi, eğitim yok) ---
        # ... (evaluate_model_pytorch_v6 kullanılır) ...
        # ... (En iyi genom JSON olarak kaydedilir) ...
        pass

        # --- Sonuçları Kaydet (v6 Final formatında, version='v6_final') ---
        # ... (final_results dict'i oluşturulur) ...
        # ... (JSON olarak kaydedilir) ...
        pass

     except (Exception, KeyboardInterrupt) as e:
          # ... (Ana hata yakalama ve W&B bitirme) ...
          pass
     finally:
          # ... (W&B normal bitirme) ...
          pass
          logging.info(f"========== PyTorch v6 Final Pipeline Run Finished ==========")


# --- Argüman Ayrıştırıcı (v6 Final) ---
def parse_arguments_v6_final() -> argparse.Namespace:
     parser = argparse.ArgumentParser(description="EvoNet v6 (Final): Genotype + Speciation + Crossover with PyTorch")
     # --- Önceki tüm gruplar (Dizin, Kontrol, Veri, Paralellik, Temel Evrim, Mutasyon, Adaptasyon, W&B, Türleşme) ---
     # v6 Adım 2 / v5'ten argümanları buraya ekle...

     # --- Çaprazlama Parametreleri Grubu ---
     cross_group = parser.add_argument_group('Crossover Parameters')
     cross_group.add_argument('--crossover_rate', type=float, default=DEFAULT_CROSSOVER_RATE, help='Probability of applying crossover instead of mutation-only reproduction.')
     cross_group.add_argument('--mate_by_avg', action=argparse.BooleanOptionalAction, default=DEFAULT_MATE_BY_AVERAGING, help='Average weights of matching genes during crossover (default: random choice).')
     # cross_group.add_argument('--interspecies_mate_rate', type=float, default=DEFAULT_INTERSPECIES_MATE_RATE, help='Probability of mating between different species (0=disabled).') # İleri seviye

     # ŞİMDİLİK YER TUTUCU - TÜM ARGÜMANLAR EKLENMELİ
     print("WARNING: Argument parser v6_final needs completion.")
     args = parser.parse_args() # Hata verir
     if args.seed is None: args.seed = random.randint(0, 2**32 - 1)
     return args

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # cli_args = parse_arguments_v6_final() # Tamamlanmış parser çağrılmalı
    # run_pipeline_pytorch_v6_final(cli_args)
    print("\nERROR: EvoNet v6 (Step 3: Crossover) code structure provided.")
    print("ERROR: Requires completion of Argument Parser and integration of Step 1/2 + v5 helper functions.")