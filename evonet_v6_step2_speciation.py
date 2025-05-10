# ==============================================================================
# EvoNet Optimizer - v6 (Adım 2: Türleşme Mekanizması)
# Açıklama: v6 Adım 1 (Genom yapısı) üzerine inşa edilmiştir. Türleşme ekler:
#           - Uyumluluk Mesafesi hesaplama
#           - Tür (Species) sınıfı ve yönetimi
#           - Fitness Paylaşımı (Adjusted Fitness)
#           - Tür bazlı üreme (henüz çaprazlama yok)
#           - Durağan türlerin elenmesi
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

# Opsiyonel W&B importu (v6 Adım 1'deki gibi)
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# --- Sabitler ve Varsayılan Değerler (v6 Adım 2 için güncellemeler) ---
# ... (Adım 1'deki sabitler: POP_SIZE, GENERATIONS, mutasyon oranları vb.) ...
DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "evonet_runs_v6_pytorch") # Aynı klasör kullanılabilir
DEFAULT_CHECKPOINT_INTERVAL = 10 # v6 için Checkpoint aralığı
# Türleşme Parametreleri
DEFAULT_COMPATIBILITY_THRESHOLD = 3.0 # Uyumluluk eşiği (NEAT'te yaygın)
DEFAULT_C1_EXCESS = 1.0       # Uyumluluk katsayısı (fazla genler)
DEFAULT_C2_DISJOINT = 1.0     # Uyumluluk katsayısı (ayrık genler)
DEFAULT_C3_WEIGHT = 0.4       # Uyumluluk katsayısı (ağırlık farkları)
DEFAULT_SPECIES_STAGNATION_LIMIT = 15 # Türün iyileşme göstermeden kalabileceği nesil sayısı

# --- Loglama, Cihaz Ayarları, Veri Üretimi (Adım 1'den aynı) ---
# setup_logging, setup_device, generate_data fonksiyonları değişmedi.
# (Kodda yer kaplamaması için tekrar eklemiyorum)
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    # ... (Adım 1 ile aynı, log dosya adı v6 olabilir) ...
    log_filename = os.path.join(log_dir, 'evolution_run_pytorch_v6.log')
    # ... (kalanı aynı) ...

# --- v6 Genom Yapısı (Adım 1'den aynı) ---
class NodeType(Enum): INPUT = auto(); OUTPUT = auto(); HIDDEN = auto()
class NodeGene: # ... (Adım 1 ile aynı) ...
    pass
class ConnectionGene: # ... (Adım 1 ile aynı) ...
    pass
class InnovationTracker: # ... (Adım 1 ile aynı) ...
    pass
class Genome: # Adım 1'den gelen Genome sınıfı
    _genome_counter = 0
    def __init__(self, genome_id: Optional[int] = None):
        self.id = genome_id if genome_id is not None else Genome._genome_counter
        Genome._genome_counter += 1
        self.node_genes: Dict[int, NodeGene] = {}
        self.connection_genes: Dict[int, ConnectionGene] = {}
        self.fitness: Optional[float] = None           # Orijinal (ham) fitness
        self.adjusted_fitness: Optional[float] = None # Türleşme sonrası ayarlanan fitness
        self.species_id: Optional[int] = None        # Ait olduğu türün ID'si

    # ... (Adım 1'deki add_node_gene, add_connection_gene, mutate metodları) ...
    # ... (Adım 1'deki mutasyon alt fonksiyonları: mutate_activation, mutate_bias, vb.) ...
    # ... (Adım 1'deki get_phenotype_model metodu) ...
    # ... (Adım 1'deki copy, __repr__, to_dict, from_dict metodları) ...
    # ÖNEMLİ: `copy` metodu adjusted_fitness ve species_id'yi None yapmalı.
    def copy(self) -> 'Genome':
        new_genome = Genome(self.id)
        new_genome.node_genes = {nid: node.copy() for nid, node in self.node_genes.items()}
        new_genome.connection_genes = {innov: conn.copy() for innov, conn in self.connection_genes.items()}
        new_genome.fitness = self.fitness # Orijinal fitness kopyalanabilir, sonraki nesilde hesaplanacak
        new_genome.adjusted_fitness = None # Ayarlanmış fitness sıfırlanır
        new_genome.species_id = None       # Tür ID'si sıfırlanır
        return new_genome

# --- v6 Fenotip Modeli (Adım 1'den aynı) ---
def _sigmoid(x): pass # ... (Adım 1 ile aynı) ...
def _relu(x): pass
def _tanh(x): pass
def _identity(x): pass
ACTIVATION_FUNCTIONS = { # ... (Adım 1 ile aynı) ...
    }
class FeedForwardNetwork(nn.Module): # ... (Adım 1 ile aynı) ...
    pass

# --- v6 Adım 2: Uyumluluk Mesafesi ---
def calculate_compatibility_distance(genome1: Genome, genome2: Genome,
                                     c1: float, c2: float, c3: float) -> float:
    """ İki genom arasındaki uyumluluk mesafesini hesaplar (NEAT formülü). """
    g1_innovs = set(genome1.connection_genes.keys())
    g2_innovs = set(genome2.connection_genes.keys())

    max_innov = max(max(g1_innovs) if g1_innovs else 0, max(g2_innovs) if g2_innovs else 0)

    excess_genes = 0
    disjoint_genes = 0
    matching_genes = 0
    weight_diff_sum = 0.0

    # İnovasyon numaralarına göre hizalama
    for innov in range(max_innov + 1):
        conn1 = genome1.connection_genes.get(innov)
        conn2 = genome2.connection_genes.get(innov)

        if conn1 is None and conn2 is not None: # Sadece g2'de var
            if innov <= max(g1_innovs) if g1_innovs else 0: # Eğer g1'in aralığındaysa disjoint
                disjoint_genes += 1
            else: # g1'in aralığının dışındaysa excess
                excess_genes += 1
        elif conn1 is not None and conn2 is None: # Sadece g1'de var
            if innov <= max(g2_innovs) if g2_innovs else 0: # Eğer g2'nin aralığındaysa disjoint
                disjoint_genes += 1
            else: # g2'nin aralığının dışındaysa excess
                excess_genes += 1
        elif conn1 is not None and conn2 is not None: # İkisinde de var (eşleşen)
            matching_genes += 1
            weight_diff_sum += abs(conn1.weight - conn2.weight)

    # Normalizasyon faktörü (genom boyutu) - Genellikle büyük genomdaki gen sayısı
    N = max(len(g1_innovs), len(g2_innovs))
    if N < 1: N = 1 # Çok küçük genomlar için (veya 20'den küçükse N=1 alınır NEAT'te)

    # Ortalama ağırlık farkı
    avg_weight_diff = weight_diff_sum / matching_genes if matching_genes > 0 else 0.0

    # Mesafe formülü
    distance = (c1 * excess_genes / N) + (c2 * disjoint_genes / N) + (c3 * avg_weight_diff)
    return distance

# --- v6 Adım 2: Tür (Species) Sınıfı ---
class Species:
    """ Genom popülasyonundaki bir türü temsil eder. """
    _species_counter = 0
    def __init__(self, representative: Genome):
        self.id = Species._species_counter
        Species._species_counter += 1
        self.representative = representative.copy() # Temsilciyi kopyala
        self.members: List[Genome] = [representative] # Başlangıçta sadece temsilci
        self.best_fitness: float = representative.fitness if representative.fitness is not None else -np.inf
        self.generations_since_improvement: int = 0
        self.offspring_to_produce: int = 0 # Bu nesil üreteceği yavru sayısı

    def add_member(self, genome: Genome):
        """ Türe yeni bir üye ekler. """
        self.members.append(genome)
        genome.species_id = self.id # Genomun tür ID'sini set et

    def update_representative(self):
        """ Türün temsilcisini rastgele bir üye ile günceller (opsiyonel). """
        if self.members:
            self.representative = random.choice(self.members).copy()

    def update_stats(self):
         """ Türün en iyi fitness'ını ve durağanlık sayacını günceller. """
         current_best_fitness_in_species = -np.inf
         for member in self.members:
              if member.fitness is not None and np.isfinite(member.fitness) and member.fitness > current_best_fitness_in_species:
                   current_best_fitness_in_species = member.fitness

         if np.isfinite(current_best_fitness_in_species) and current_best_fitness_in_species > self.best_fitness:
              self.best_fitness = current_best_fitness_in_species
              self.generations_since_improvement = 0
              logging.debug(f"Species {self.id} improved. New best fitness: {self.best_fitness:.6f}")
         else:
              self.generations_since_improvement += 1
              logging.debug(f"Species {self.id} did not improve. Stagnation: {self.generations_since_improvement}")


    def adjust_fitnesses(self):
        """ Fitness paylaşımını uygular (explicit fitness sharing). """
        num_members = len(self.members)
        if num_members == 0: return

        for member in self.members:
            if member.fitness is not None and np.isfinite(member.fitness):
                member.adjusted_fitness = member.fitness / num_members
            else:
                member.adjusted_fitness = -np.inf # Geçersiz fitness'ı olanları cezalandır

    def calculate_offspring_count(self, total_adjusted_fitness: float, pop_size: int, elitism_count: int) -> float:
        """ Bu türün üretmesi gereken yavru sayısını hesaplar. """
        if total_adjusted_fitness <= 0: return 0.0 # Bölme hatasını önle

        species_adjusted_fitness_sum = sum(m.adjusted_fitness for m in self.members if m.adjusted_fitness is not None and np.isfinite(m.adjusted_fitness))
        proportion = species_adjusted_fitness_sum / total_adjusted_fitness
        num_offspring = proportion * (pop_size - elitism_count) # Elitler dışındaki pay
        return num_offspring

    def select_parent(self) -> Genome:
         """ Tür içinden üreme için bir ebeveyn seçer (turnuva veya rulet).
             Ayarlanmış fitness'a göre seçmek mantıklıdır.
         """
         # Basit Turnuva Seçimi (Ayarlanmış Fitness'a göre)
         k = min(5, len(self.members)) # Küçük turnuva boyutu
         if k <= 0:
              # Bu durum olmamalı ama güvenlik için
              if self.members: return random.choice(self.members)
              else: raise ValueError(f"Species {self.id} has no members to select parent from.")

         tournament_members = random.sample(self.members, k)
         winner = max(tournament_members, key=lambda g: g.adjusted_fitness if g.adjusted_fitness is not None else -np.inf)
         return winner


    def reset(self):
        """ Yeni nesil için türü sıfırlar (üyeleri temizler). Temsilci kalır. """
        # Temsilciyi güncellemek iyi bir fikir olabilir
        self.update_representative()
        self.members = [] # Üyeleri temizle
        self.offspring_to_produce = 0 # Yavru sayısını sıfırla

    def __repr__(self) -> str:
        return f"Species(id={self.id}, members={len(self.members)}, best_fit={self.best_fitness:.4f}, stagnated={self.generations_since_improvement})"


# --- v6 Adım 2: Evrim Döngüsü (Türleşme ile) ---
def evolve_population_pytorch_v6_speciation( # Fonksiyon adı güncellendi
    population: List[Genome],
    X_train_np: np.ndarray, y_train_np: np.ndarray,
    start_generation: int, total_generations: int,
    args: argparse.Namespace,
    output_dir: str, device: torch.device,
    innovation_tracker: InnovationTracker,
    wandb_run: Optional[Any]
) -> Tuple[Optional[Genome], List[float], List[float]]:
    """ PyTorch v6 Genom tabanlı evrimsel süreci TÜRLEŞME ile çalıştırır. """

    best_fitness_history = [] # Orijinal en iyi fitness'ı takip eder
    avg_fitness_history = []  # Orijinal ortalama fitness'ı takip eder
    best_genome_overall: Optional[Genome] = None
    best_fitness_overall = -np.inf

    current_mutation_strength = args.mutation_strength
    stagnation_counter_global = 0 # Global tıkanma (tüm türler için)

    pop_size = len(population)
    fitness_params = {'complexity_penalty': args.complexity_penalty}

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) if args.num_workers > 0 else None
    if executor: logging.info(f"v6 Speciation: Using ProcessPoolExecutor with {args.num_workers} workers.")

    species_list: List[Species] = [] # Aktif türlerin listesi

    try:
        for gen in range(start_generation, total_generations):
            generation_start_time = time.time()

            # --- 1. Fitness Değerlendirme (Adım 1'deki gibi Paralel/Seri) ---
            fitness_scores = [-np.inf] * pop_size
            phenotype_creation_errors = [False] * pop_size
            # ... (Adım 1'deki fitness hesaplama kodu buraya gelir - _calculate_fitness_worker_v6 kullanılır) ...
            # ... (Hesaplama sonucunda fitness_scores dolar ve genomların .fitness'ı güncellenir) ...
            # >>> Buraya Adım 1'deki fitness hesaplama bloğunu ekleyin <<<
            # Örnek (Seri):
            if not executor:
                 temp_device = torch.device("cpu")
                 for i, genome in enumerate(population):
                      fitness = calculate_fitness_pytorch_v6(genome, X_train_np, y_train_np, temp_device, fitness_params)
                      if fitness is None: fitness_scores[i] = -np.inf; phenotype_creation_errors[i] = True
                      else: fitness_scores[i] = fitness; genome.fitness = fitness
            else:
                 # ... Adım 1'deki paralel fitness hesaplama bloğu ...
                 pass # Yukarıdaki koddan alınmalı


            num_phenotype_errors = sum(phenotype_creation_errors)
            # ... (Adım 1'deki istatistikler, en iyiyi takip - ORİJİNAL fitness ile yapılır) ...
            valid_indices = [i for i, score in enumerate(fitness_scores) if np.isfinite(score)]
            if not valid_indices: raise RuntimeError(f"Gen {gen+1}: No valid individuals!") # Hata ver
            # ... (current_best_fitness, avg_fitness, best_genome_overall takibi - Adım 1'deki gibi) ...
            # >>> Buraya Adım 1'deki istatistik/takip bloğunu ekleyin <<<


            # --- 2. Türleştirme (Speciation) ---
            logging.debug(f"Generation {gen+1}: Starting speciation...")
            # Önceki nesilden kalan türlerin üyelerini temizle (temsilciler kalır)
            for s in species_list: s.reset()

            newly_created_species_count = 0
            assigned_genomes = 0
            for i, genome in enumerate(population):
                if genome.fitness is None or not np.isfinite(genome.fitness): continue # Geçersiz genomları atla

                found_species = False
                for s in species_list:
                    dist = calculate_compatibility_distance(genome, s.representative, args.c1_excess, args.c2_disjoint, args.c3_weight)
                    if dist < args.compatibility_threshold:
                        s.add_member(genome)
                        found_species = True
                        assigned_genomes += 1
                        break # İlk uygun türe ata

                if not found_species:
                    # Yeni tür oluştur
                    new_species = Species(genome)
                    species_list.append(new_species)
                    newly_created_species_count += 1
                    assigned_genomes += 1

            logging.info(f"Speciation complete: {len(species_list)} species ({newly_created_species_count} new). {assigned_genomes}/{pop_size} genomes assigned.")

            # Boş kalan türleri temizle (eğer reset sonrası hiç üye atanmadıysa)
            species_list = [s for s in species_list if s.members]
            if not species_list:
                 logging.error("No species left after speciation! Evolution cannot continue.")
                 raise RuntimeError("All species died out during speciation.")
            logging.debug(f"Removed empty species. {len(species_list)} species remaining.")


            # --- 3. Fitness Paylaşımı ve İstatistik Güncelleme ---
            total_adjusted_fitness = 0.0
            for s in species_list:
                s.adjust_fitnesses() # Fitness'ı tür boyutuna göre ayarla
                s.update_stats()     # En iyi fitness ve durağanlık sayacını güncelle
                # Toplam ayarlanmış fitness'ı hesapla (yavru sayısı için)
                total_adjusted_fitness += sum(m.adjusted_fitness for m in s.members if m.adjusted_fitness is not None and np.isfinite(m.adjusted_fitness))

            logging.debug(f"Fitness sharing applied. Total adjusted fitness: {total_adjusted_fitness:.4f}")


            # --- 4. Durağan Türleri Eleme ve Yavru Sayısını Hesaplama ---
            pop_to_fill = pop_size - args.elitism_count # Elitler hariç doldurulacak yer
            offspring_counts_float: List[Tuple[float, Species]] = [] # (sayı, tür)

            if total_adjusted_fitness <= 0:
                 logging.warning("Total adjusted fitness is zero or negative. Assigning equal offspring counts.")
                 # Eşit dağıt veya rastgele yap? Eşit dağıtalım.
                 equal_share = pop_to_fill / len(species_list) if species_list else 0
                 for s in species_list: offspring_counts_float.append((equal_share, s))
            else:
                 for s in species_list:
                     # Durağanlık kontrolü (elitizmi koruyarak)
                     is_stagnant = s.generations_since_improvement > args.species_stagnation_limit
                     # Çok az tür kaldıysa veya en iyi türe sahipse eleme (koruma)
                     should_protect = len(species_list) <= 2 or (best_genome_overall and best_genome_overall.species_id == s.id)

                     if is_stagnant and not should_protect:
                          logging.info(f"Species {s.id} removed due to stagnation ({s.generations_since_improvement} generations).")
                          # Bu türe yavru hakkı verme (listeden çıkar)
                     else:
                          offspring_f = s.calculate_offspring_count(total_adjusted_fitness, pop_size, args.elitism_count)
                          offspring_counts_float.append((offspring_f, s))

            # Türler elendikten sonra listeyi güncelle
            species_list = [item[1] for item in offspring_counts_float]
            if not species_list:
                 logging.error("No species left after stagnation removal! Evolution cannot continue.")
                 raise RuntimeError("All species died out after stagnation removal.")

            # Yavru sayılarını tam sayıya çevir (yuvarlama hatalarını yöneterek)
            total_offspring_needed = pop_to_fill
            assigned_offspring = 0
            offspring_counts_int: List[Tuple[int, Species]] = []

            # Önce tam kısımları ata
            for count_f, s in offspring_counts_float:
                count_i = int(count_f)
                offspring_counts_int.append((count_i, s))
                assigned_offspring += count_i

            # Kalanları, ondalık kısımlara göre en büyüklerden başlayarak ata
            remaining_offspring = total_offspring_needed - assigned_offspring
            if remaining_offspring > 0:
                 fractions = sorted([(count_f - int(count_f), i) for i, (count_f, s) in enumerate(offspring_counts_float)], reverse=True)
                 for _, index in fractions[:remaining_offspring]:
                      current_count, species_ref = offspring_counts_int[index]
                      offspring_counts_int[index] = (current_count + 1, species_ref)
                      assigned_offspring += 1

            # Güvenlik kontrolü
            if assigned_offspring != total_offspring_needed:
                 logging.warning(f"Offspring count mismatch! Needed {total_offspring_needed}, assigned {assigned_offspring}. Adjusting...")
                 # Farkı rastgele bir türe ekle/çıkar (basit çözüm)
                 diff = total_offspring_needed - assigned_offspring
                 if offspring_counts_int:
                      idx_to_adjust = random.randrange(len(offspring_counts_int))
                      count, s = offspring_counts_int[idx_to_adjust]
                      offspring_counts_int[idx_to_adjust] = (max(0, count + diff), s)


            # Her türe atanmış yavru sayısını kaydet
            final_offspring_map = {s.id: count for count, s in offspring_counts_int}
            for s in species_list:
                 s.offspring_to_produce = final_offspring_map.get(s.id, 0)
                 logging.debug(f"Species {s.id} assigned {s.offspring_to_produce} offspring.")

            # --- 5. Yeni Popülasyon Oluşturma (Tür Bazlı Üreme) ---
            new_population = []

            # 5a. Elitizm (Genel en iyileri - Orijinal fitness'a göre)
            if args.elitism_count > 0:
                try:
                    # Tüm popülasyondaki geçerli bireyleri orijinal fitness'a göre sırala
                    sorted_overall_indices = sorted(valid_indices, key=lambda i: population[i].fitness, reverse=True)
                    elite_indices = sorted_overall_indices[:args.elitism_count]
                    for idx in elite_indices:
                        elite_clone = population[idx].copy()
                        new_population.append(elite_clone)
                    logging.debug(f"Added {len(new_population)} overall elites.")
                except Exception as e: logging.error(f"Error during overall elitism: {e}", exc_info=True)

            # 5b. Yavruları Üretme (Tür İçinden Seçim + Mutasyon)
            generated_offspring = 0
            for s in species_list:
                if not s.members: continue # Üyesi olmayan türden üretim yapılamaz
                for _ in range(s.offspring_to_produce):
                     if generated_offspring >= pop_to_fill: break # Gerekli sayıya ulaşıldıysa dur
                     try:
                          # Tür içinden ebeveyn seç (ayarlanmış fitness'a göre)
                          parent_genome = s.select_parent()
                          # Kopyala ve mutasyona uğrat
                          child_genome = parent_genome.copy()
                          # Adaptif gücü argümanlara ekle (mutate kullanabilsin diye)
                          args.current_mutation_strength = current_mutation_strength
                          child_genome.mutate(innovation_tracker, args)
                          del args.current_mutation_strength

                          new_population.append(child_genome)
                          generated_offspring += 1
                     except Exception as e:
                          logging.error(f"Error producing offspring for species {s.id}: {e}", exc_info=True)
                if generated_offspring >= pop_to_fill: break

            # Eksik kalırsa (çok nadir olmalı)
            if len(new_population) < pop_size:
                 logging.warning(f"Population size shortfall ({len(new_population)}/{pop_size}). Filling with random mutants from best species.")
                 # En iyi türden rastgele bireyler alıp mutasyona uğratarak doldur
                 best_species = max(species_list, key=lambda s: s.best_fitness, default=None)
                 if best_species and best_species.members:
                      while len(new_population) < pop_size:
                           parent = random.choice(best_species.members).copy()
                           args.current_mutation_strength = current_mutation_strength
                           parent.mutate(innovation_tracker, args)
                           del args.current_mutation_strength
                           new_population.append(parent)
                 else: # Hiç tür yoksa minimal genom ekle
                      logging.error("Cannot fill population: No species available.")
                      # while len(new_population) < pop_size: new_population.append(create_initial_genome(...))


            population = new_population[:pop_size] # Boyutu garantile

            # --- 6. Checkpoint Alma (v6 - Tür durumu da kaydedilebilir) ---
            if args.checkpoint_interval > 0 and (gen + 1) % args.checkpoint_interval == 0:
                 try:
                     rnd_state = random.getstate(); np_rnd_state = np.random.get_state(); torch_rnd_state = torch.get_rng_state().cpu()
                     tracker_state = innovation_tracker.get_state()
                     wandb_id = wandb_run.id if wandb_run else None
                     # Türlerin durumunu da kaydetmek gerekebilir (ID, representative, stats)
                     species_state = [{'id': s.id, 'rep': s.representative.to_dict(), 'best_fit': s.best_fitness, 'stagnated': s.generations_since_improvement} for s in species_list]
                     save_checkpoint_pytorch_v6_speciation(output_dir, gen + 1, population, tracker_state, species_state,
                                                          rnd_state, np_rnd_state, torch_rnd_state, wandb_id) # Yeni kaydetme fonksiyonu
                 except Exception as e:
                     logging.error(f"Failed checkpoint saving for gen {gen+1}: {e}", exc_info=True)

    finally: # Executor'ı kapat
        if executor: executor.shutdown(wait=True)

    # Evrim Sonu (En iyi genomu seç - Orijinal fitness'a göre)
    if best_genome_overall is None and population:
         # ... (Adım 1'deki final seçim kodu - .fitness kullanılır) ...
         pass # Yukarıdaki kodu buraya yapıştırın
    # ... (Adım 1'deki diğer final durumlar) ...

    return best_genome_overall, best_fitness_history, avg_fitness_history


# --- v6 Adım 2: Checkpointing (Tür Durumu ile) ---
def save_checkpoint_pytorch_v6_speciation(output_dir: str, generation: int, population: List[Genome],
                                          tracker_state: Dict[str, int], species_state: List[Dict], # Tür durumu eklendi
                                          rnd_state: Any, np_rnd_state: Any, torch_rnd_state: Any,
                                          wandb_run_id: Optional[str] = None):
    """ Evrim durumunu (PyTorch v6 - Genomlar, Türler) kaydeder. """
    checkpoint_dir = os.path.join(output_dir, "checkpoints_pytorch_v6") # Aynı klasör
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"evo_gen_{generation}.pt")
    logging.info(f"Saving v6_speciation checkpoint for gen {generation} to {checkpoint_file}...")
    population_state_dicts = [g.to_dict() for g in population] # Hata kontrolü eklenebilir
    state = {
        "version": "v6_speciation", # Sürüm detayı
        "generation": generation,
        "population_state": population_state_dicts,
        "innovation_tracker_state": tracker_state,
        "species_state": species_state, # Kaydedilen tür bilgisi
        "random_state": rnd_state, "numpy_random_state": np_rnd_state, "torch_random_state": torch_rnd_state,
        "wandb_run_id": wandb_run_id, "timestamp": datetime.now().isoformat()
    }
    try: torch.save(state, checkpoint_file); logging.info(f"v6 Checkpoint saved.")
    except Exception as e: logging.error(f"Failed to save v6 checkpoint: {e}", exc_info=True)

# --- v6 Adım 2: Checkpoint Yükleme (Tür Durumu ile) ---
def load_checkpoint_pytorch_v6_speciation(checkpoint_path: str) -> Optional[Dict]:
    """ Kaydedilmiş PyTorch v6 evrim durumunu (Genomlar, Türler) yükler. """
    if not os.path.exists(checkpoint_path): logging.error(f"Checkpoint file not found: {checkpoint_path}"); return None
    logging.info(f"Loading v6_speciation checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # if "v6_speciation" not in checkpoint.get("version", ""): logging.warning(...)

        population = [Genome.from_dict(g_dict) for g_dict in checkpoint["population_state"]] # Hata kontrolü eklenebilir
        if not population: logging.error("Failed to load any genome."); return None

        # Tür durumunu yükle
        loaded_species = []
        if "species_state" in checkpoint:
             for s_state in checkpoint["species_state"]:
                  try:
                       rep_genome = Genome.from_dict(s_state['rep'])
                       species = Species(rep_genome) # Yeni tür objesi oluştur
                       species.id = s_state['id'] # ID'yi geri yükle
                       species.best_fitness = s_state['best_fit']
                       species.generations_since_improvement = s_state['stagnated']
                       loaded_species.append(species)
                  except Exception as e: logging.error(f"Failed to load species state (ID: {s_state.get('id')}): {e}")
             logging.info(f"Loaded state for {len(loaded_species)} species.")

        checkpoint["population"] = population
        checkpoint["species_list"] = loaded_species # Yüklenen tür listesini ekle
        checkpoint["innovation_tracker_state"] = checkpoint.get("innovation_tracker_state")
        checkpoint["wandb_run_id"] = checkpoint.get("wandb_run_id")
        logging.info(f"Checkpoint loaded. Resuming from gen {checkpoint['generation'] + 1}.")
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load v6 checkpoint: {e}", exc_info=True)
        return None

# --- Ana İş Akışı (v6 - Türleşme ile) ---
def run_pipeline_pytorch_v6_speciation(args: argparse.Namespace):
     wandb_run = None; output_dir = None; species_list: List[Species] = [] # Tür listesi
     try:
        # ... (v6 Adım 1'deki başlangıç ayarları: device, timestamp, output_dir, logging) ...
        # >>> Buraya Adım 1 / v5'teki başlangıç kodunu ekleyin <<<

        # Checkpoint Yükleme (v6 - Türleşme ile)
        start_generation = 0; population: List[Genome] = []; initial_state_loaded = False; tracker_state = None; resumed_wandb_id = None
        latest_checkpoint_path = find_latest_checkpoint_pytorch(output_dir) if resume_run else None

        if latest_checkpoint_path:
            loaded_state = load_checkpoint_pytorch_v6_speciation(latest_checkpoint_path) # Yeni yükleyici
            if loaded_state:
                start_generation = loaded_state['generation']
                population = loaded_state['population']
                species_list = loaded_state.get('species_list', []) # Yüklenen tür listesi
                tracker_state = loaded_state.get("innovation_tracker_state")
                resumed_wandb_id = loaded_state.get("wandb_run_id")
                # ... (v6 Adım 1'deki random state yükleme) ...
                initial_state_loaded = True
                logging.info(f"Resuming from Gen {start_generation + 1} with {len(population)} genomes and {len(species_list)} species structures.")
            # ... (v6 Adım 1'deki checkpoint yükleme başarısızlık durumu) ...
        # ... (v6 Adım 1'deki resume=True ama checkpoint yok durumu) ...

        # Innovation Tracker Başlatma/Yükleme
        innovation_tracker = InnovationTracker()
        if initial_state_loaded and tracker_state: innovation_tracker.set_state(tracker_state); logging.info("Tracker state restored.")
        else: logging.info("Initialized new innovation tracker.")

        # W&B Başlatma (v5'teki gibi)
        # ...

        # Config Kaydetme/Loglama (v5'teki gibi)
        # ...

        # Random Tohum Ayarlama (sadece sıfırdan başlarken)
        # ...

        # Veri Üretimi (v5'teki gibi)
        # ... (input_size, output_size alınır) ...

        # Popülasyon Başlatma (sadece sıfırdan başlarken)
        if not initial_state_loaded:
            logging.info(f"--- Initializing Population (Size: {args.pop_size}) with Minimal Genomes (v6 Speciation) ---")
            try:
                population = [create_initial_genome(input_size, output_size, innovation_tracker) for _ in range(args.pop_size)]
                logging.info("Population initialized.")
                # İlk nesilde türleşme hemen yapılır mı? Evet, evrim döngüsü içinde yapılacak.
            except Exception: logging.critical("Failed to initialize population."); sys.exit(1)

        # Evrim Süreci (v6 - Türleşme ile)
        logging.info(f"--- Starting/Resuming PyTorch v6 Evolution with Speciation ---")
        best_genome_evolved: Optional[Genome] = None
        best_fitness_hist = [] # TODO: yükle
        avg_fitness_hist = []  # TODO: yükle

        if start_generation >= args.generations:
            # ... (v6 Adım 1'deki gibi evrim atlama ve en iyi genomu seçme) ...
            pass # Önceki kodu buraya yapıştırın
        else:
             try:
                 # TÜRLEŞME içeren evrim fonksiyonunu çağır
                 best_genome_evolved, gen_best_hist, gen_avg_hist = evolve_population_pytorch_v6_speciation(
                     population, X_train_np, y_train_np, start_generation, args.generations,
                     args, output_dir, device, innovation_tracker, wandb_run,
                     species_list # Yüklenen tür listesini de verelim (gerçi döngüde sıfırlanacak ama ilk nesil için?)
                     # Düzeltme: Evrim fonksiyonuna species_list'i vermek yerine, fonksiyon içinde yönetilsin.
                     # Fonksiyonun başlangıcında, eğer start_generation > 0 ise yüklenen listeyi kullanır, değilse boş başlatır.
                 )
                 best_fitness_hist.extend(gen_best_hist)
                 avg_fitness_hist.extend(gen_avg_hist)
             except Exception as e:
                 logging.critical(f"Fatal error during PyTorch v6 speciation evolution: {e}", exc_info=True)
                 raise

        logging.info("--- PyTorch v6 Speciation Evolution Complete ---")

        # Fitness Geçmişi Kaydet/Çizdir (v5'teki gibi, dosya adı v6 olabilir)
        # ...

        # En İyi Genomu Değerlendir/Kaydet (v6 Adım 1'deki gibi, eğitim yok)
        # ...

        # Sonuçları Kaydet (v6 formatında, version='v6_speciation')
        # ...

     except (Exception, KeyboardInterrupt) as e:
          # ... (v5'teki ana hata yakalama ve W&B bitirme) ...
          pass
     finally:
          # ... (v5'teki W&B normal bitirme) ...
          pass
          logging.info(f"========== PyTorch v6 Speciation Pipeline Run Finished ==========")

# --- Argüman Ayrıştırıcı (v6 - Türleşme ile) ---
def parse_arguments_v6_speciation() -> argparse.Namespace:
     parser = argparse.ArgumentParser(description="EvoNet v6 (Speciation): Genotype + Speciation with PyTorch")
     # --- Önceki gruplar (Dizin, Kontrol, Veri, Paralellik, Temel Evrim, Mutasyon, Adaptasyon, W&B) ---
     # v6 Adım 1 / v5'ten argümanları buraya ekle...

     # --- Türleşme Parametreleri Grubu ---
     spec_group = parser.add_argument_group('Speciation Parameters')
     spec_group.add_argument('--compatibility_threshold', type=float, default=DEFAULT_COMPATIBILITY_THRESHOLD, help='Distance threshold for species compatibility.')
     spec_group.add_argument('--c1_excess', type=float, default=DEFAULT_C1_EXCESS, help='Compatibility coefficient for excess genes.')
     spec_group.add_argument('--c2_disjoint', type=float, default=DEFAULT_C2_DISJOINT, help='Compatibility coefficient for disjoint genes.')
     spec_group.add_argument('--c3_weight', type=float, default=DEFAULT_C3_WEIGHT, help='Compatibility coefficient for weight differences.')
     spec_group.add_argument('--species_stagnation_limit', type=int, default=DEFAULT_SPECIES_STAGNATION_LIMIT, help='Generations a species can stagnate before removal.')

     # ŞİMDİLİK YER TUTUCU - ÖNCEKİ TÜM ARGÜMANLAR BURAYA EKLENMELİ
     print("WARNING: Argument parser v6_speciation needs completion.")
     args = parser.parse_args()
     if args.seed is None: args.seed = random.randint(0, 2**32 - 1)
     return args

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # cli_args = parse_arguments_v6_speciation() # Tamamlanmış parser çağrılmalı
    # run_pipeline_pytorch_v6_speciation(cli_args)
    print("\nERROR: EvoNet v6 (Step 2: Speciation) code structure provided.")
    print("ERROR: Requires completion of Argument Parser and integration of Step 1 + v5 helper functions.")