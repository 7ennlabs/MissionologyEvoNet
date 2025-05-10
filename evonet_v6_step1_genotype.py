# ==============================================================================
# EvoNet Optimizer - v6 (Adım 1: Genom ve Yapısal Mutasyon Temelleri)
# Açıklama: v5 üzerine inşa edilmiştir. Ağları temsil etmek için Genom yapısı
#           (NodeGene, ConnectionGene), küresel inovasyon takibi, temel yapısal
#           mutasyonlar (bağlantı ekle, düğüm ekle) ve genomdan basit
#           ileri beslemeli fenotip (PyTorch modeli) oluşturma eklenmiştir.
#           Türleşme ve gelişmiş çaprazlama henüz eklenmemiştir.
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
import math # sigmoid için

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Opsiyonel W&B importu (v5'teki gibi)
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    # print("Warning: wandb library not found...") # Tekrar yazdırmaya gerek yok

# --- Sabitler ve Varsayılan Değerler (v6 için güncellemeler) ---
# ... (v5'teki temel sabitler kalabilir: POP_SIZE, GENERATIONS, vb.) ...
DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "evonet_runs_v6_pytorch")
# Mutasyon Oranları (v6) - Artık daha fazla mutasyon türü var
DEFAULT_CONN_MUT_RATE = 0.80  # Mevcut bağlantıların ağırlıklarını değiştirme olasılığı
DEFAULT_BIAS_MUT_RATE = 0.70  # Mevcut düğümlerin bias'larını değiştirme olasılığı
DEFAULT_ADD_CONN_RATE = 0.10  # Yeni bağlantı ekleme olasılığı
DEFAULT_ADD_NODE_RATE = 0.05  # Yeni düğüm ekleme olasılığı
DEFAULT_TOGGLE_ENABLE_RATE = 0.05 # Bağlantıyı etkinleştirme/devre dışı bırakma olasılığı
DEFAULT_ACTIVATION_MUT_RATE = 0.02 # Aktivasyon fonksiyonunu değiştirme olasılığı
# Not: Bu oranların toplamı 1'den büyük olabilir, her biri bağımsız kontrol edilir.

# Diğer v5 sabitleri (varsayılanları ayarlanabilir)
DEFAULT_MUTATION_STRENGTH = 0.1 # Ağırlık/Bias mutasyon gücü
DEFAULT_COMPLEXITY_PENALTY = 0.00001
DEFAULT_NUM_WORKERS = 0
DEFAULT_ADAPT_MUTATION = True # Adaptif mutasyon gücü hala kullanılabilir (ağırlık/bias için)
# ... (Diğer adaptif mutasyon ve v5 sabitleri) ...

AVAILABLE_ACTIVATIONS = ['sigmoid', 'relu', 'tanh', 'identity'] # Kullanılabilir aktivasyonlar

# --- Loglama, Cihaz Ayarları, Veri Üretimi (v5'ten aynı) ---
# setup_logging, setup_device, generate_data fonksiyonları değişmedi.
# (Kodda yer kaplamaması için tekrar eklemiyorum, v5'teki gibi olduklarını varsayalım)
# Sadece log dosya adı ve başlığı v6 olarak güncellenebilir.
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    log_filename = os.path.join(log_dir, 'evolution_run_pytorch_v6.log')
    for handler in logging.root.handlers[:]: handler.close(); logging.root.removeHandler(handler)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)-8s [%(filename)s:%(lineno)d] - %(message)s',
                        handlers=[logging.FileHandler(log_filename, mode='a'), logging.StreamHandler(sys.stdout)])
    logging.info("="*50); logging.info("PyTorch EvoNet v6 (Genom Temelli) Logging Başlatıldı."); logging.info("="*50)

def setup_device(requested_device: str) -> torch.device:
     # ... (v5 ile aynı) ...
     pass # Önceki kodu buraya yapıştırın

def generate_data(num_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
     # ... (v5 ile aynı) ...
     pass # Önceki kodu buraya yapıştırın


# === v6: Genom Yapısı ===

class NodeType(Enum):
    INPUT = auto()
    OUTPUT = auto()
    HIDDEN = auto()

class NodeGene:
    """ Bir nöronu (düğümü) temsil eden gen. """
    def __init__(self, id: int, node_type: NodeType, bias: float = 0.0, activation: str = 'sigmoid'):
        self.id = id
        self.type = node_type
        self.bias = np.float32(bias) # Bias'ı float32 yapalım
        if activation not in AVAILABLE_ACTIVATIONS:
            logging.warning(f"Node {id}: Unknown activation '{activation}'. Defaulting to 'sigmoid'.")
            activation = 'sigmoid'
        self.activation = activation

    def copy(self) -> 'NodeGene':
        return NodeGene(self.id, self.type, self.bias, self.activation)

    def __repr__(self) -> str:
        return f"NodeGene(id={self.id}, type={self.type.name}, bias={self.bias:.3f}, act={self.activation})"

    def to_dict(self) -> Dict[str, Any]:
        return {'id': self.id, 'type': self.type.name, 'bias': self.bias, 'activation': self.activation}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeGene':
        return cls(data['id'], NodeType[data['type']], data['bias'], data['activation'])


class ConnectionGene:
    """ İki düğüm arasındaki bağlantıyı temsil eden gen. """
    def __init__(self, in_node_id: int, out_node_id: int, weight: float, enabled: bool, innovation_number: int):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = np.float32(weight) # Ağırlığı float32 yapalım
        self.enabled = enabled
        self.innovation_number = innovation_number # Bu genin evrimdeki benzersiz kimliği

    def copy(self) -> 'ConnectionGene':
        return ConnectionGene(self.in_node_id, self.out_node_id, self.weight, self.enabled, self.innovation_number)

    def __repr__(self) -> str:
        status = "E" if self.enabled else "D"
        return f"ConnGene(in={self.in_node_id}, out={self.out_node_id}, w={self.weight:.3f}, {status}, innov={self.innovation_number})"

    def to_dict(self) -> Dict[str, Any]:
        return {'in': self.in_node_id, 'out': self.out_node_id, 'weight': self.weight,
                'enabled': self.enabled, 'innov': self.innovation_number}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConnectionGene':
        return cls(data['in'], data['out'], data['weight'], data['enabled'], data['innov'])

class InnovationTracker:
    """ Yeni genler (düğüm ID'leri ve bağlantı inovasyonları) için küresel sayaçları yönetir.
        Not: Paralel süreçlerde dikkatli kullanılmalıdır. Bu implementasyon
        mutasyonların ana süreçte yapıldığı varsayımıyla basitleştirilmiştir.
    """
    def __init__(self, initial_node_id: int = 0, initial_innovation: int = 0):
        self._node_id_counter = initial_node_id
        self._innovation_counter = initial_innovation
        # self._innovations = {} # (Gelişmiş: Aynı yapısal mutasyonun tekrarını takip etmek için)

    def get_new_node_id(self) -> int:
        new_id = self._node_id_counter
        self._node_id_counter += 1
        return new_id

    def get_new_innovation_number(self) -> int:
         # Basit sayaç. Gelişmiş versiyon, aynı yapısal mutasyon için
         # (örn: node 1'den node 3'e bağlantı ekleme) aynı numarayı döndürmelidir.
         # Bu, çaprazlama için önemlidir. Şimdilik basit tutuyoruz.
        new_innov = self._innovation_counter
        self._innovation_counter += 1
        return new_innov

    def get_state(self) -> Dict[str, int]:
        return {'node_id': self._node_id_counter, 'innovation': self._innovation_counter}

    def set_state(self, state: Dict[str, int]):
        self._node_id_counter = state.get('node_id', 0)
        self._innovation_counter = state.get('innovation', 0)

class Genome:
    """ Bir sinir ağının genetik temsilini (genotip) tutar. """
    _genome_counter = 0 # Benzersiz genom ID'leri için

    def __init__(self, genome_id: Optional[int] = None):
        self.id = genome_id if genome_id is not None else Genome._genome_counter
        Genome._genome_counter += 1
        self.node_genes: Dict[int, NodeGene] = {} # node_id -> NodeGene
        self.connection_genes: Dict[int, ConnectionGene] = {} # innovation_number -> ConnectionGene
        self.fitness: Optional[float] = None # Evrim sırasında hesaplanacak

    def add_node_gene(self, node: NodeGene):
        self.node_genes[node.id] = node

    def add_connection_gene(self, connection: ConnectionGene):
        self.connection_genes[connection.innovation_number] = connection

    def mutate(self, innovation_tracker: InnovationTracker, args: argparse.Namespace):
        """ Genoma çeşitli mutasyonları uygular. """
        # Not: Oranlar args'tan gelmeli
        # 1. Aktivasyon Mutasyonu
        if random.random() < args.activation_mut_rate:
            self.mutate_activation()
        # 2. Bias Mutasyonu
        if random.random() < args.bias_mut_rate:
            self.mutate_bias(args.mutation_strength) # Adaptif güç bias'a da uygulanabilir mi? Şimdilik sabit.
        # 3. Bağlantı Ağırlık Mutasyonu
        if random.random() < args.conn_mut_rate:
            # Adaptif gücü burada kullanalım
            adaptive_strength = get_current_adaptive_strength(args) # Bu fonksiyon dışarıda tanımlanmalı
            self.mutate_connection_weights(adaptive_strength)
        # 4. Bağlantı Etkinleştirme/Devre Dışı Bırakma
        if random.random() < args.toggle_enable_rate:
            self.mutate_toggle_enable()
        # 5. Yeni Bağlantı Ekleme
        if random.random() < args.add_conn_rate:
            self.mutate_add_connection(innovation_tracker)
        # 6. Yeni Düğüm Ekleme
        if random.random() < args.add_node_rate:
            self.mutate_add_node(innovation_tracker)

    def mutate_activation(self):
        """ Rastgele bir gizli düğümün aktivasyon fonksiyonunu değiştirir. """
        hidden_nodes = [n for n in self.node_genes.values() if n.type == NodeType.HIDDEN]
        if not hidden_nodes: return
        node_to_mutate = random.choice(hidden_nodes)
        current_activation = node_to_mutate.activation
        possible_new_activations = [act for act in AVAILABLE_ACTIVATIONS if act != current_activation]
        if possible_new_activations:
            new_activation = random.choice(possible_new_activations)
            logging.debug(f"Genome {self.id}: Mutating activation for Node {node_to_mutate.id} from {current_activation} to {new_activation}")
            node_to_mutate.activation = new_activation

    def mutate_bias(self, strength: float):
        """ Rastgele düğümlerin bias'larına gürültü ekler. """
        for node in self.node_genes.values():
            if node.type != NodeType.INPUT: # Giriş düğümlerinin bias'ı olmaz/önemsiz
                 # Her bias için ayrı olasılık mı, yoksa tek seferde mi? Tek seferde yapalım.
                 noise = np.random.normal(0, strength)
                 node.bias += np.float32(noise)
                 # logging.debug(f"Genome {self.id}: Mutating bias for Node {node.id} by {noise:.3f}")

    def mutate_connection_weights(self, strength: float):
        """ Mevcut bağlantıların ağırlıklarına gürültü ekler. """
        for conn in self.connection_genes.values():
            # Her ağırlık için ayrı olasılık mı? v5'teki weight_mut_rate gibi mi? Evet.
            if random.random() < DEFAULT_WEIGHT_MUT_RATE: # Sabit kullanalım şimdilik (veya args'tan al)
                 noise = np.random.normal(0, strength)
                 conn.weight += np.float32(noise)
                 # logging.debug(f"Genome {self.id}: Mutating weight for Conn {conn.innovation_number} by {noise:.3f}")

    def mutate_toggle_enable(self):
        """ Rastgele bir bağlantının etkin/devre dışı durumunu değiştirir. """
        if not self.connection_genes: return
        conn_to_toggle = random.choice(list(self.connection_genes.values()))
        conn_to_toggle.enabled = not conn_to_toggle.enabled
        logging.debug(f"Genome {self.id}: Toggled connection {conn_to_toggle.innovation_number} to {'enabled' if conn_to_toggle.enabled else 'disabled'}")

    def mutate_add_connection(self, innovation_tracker: InnovationTracker):
        """ İki bağlı olmayan düğüm arasına yeni bir bağlantı geni ekler. """
        possible_starts = [n.id for n in self.node_genes.values() if n.type != NodeType.OUTPUT]
        possible_ends = [n.id for n in self.node_genes.values() if n.type != NodeType.INPUT]

        if not possible_starts or not possible_ends: return

        max_attempts = 20 # Sonsuz döngüyü önle
        for _ in range(max_attempts):
            start_node_id = random.choice(possible_starts)
            end_node_id = random.choice(possible_ends)

            # Aynı düğüme bağlantı yok
            if start_node_id == end_node_id: continue
            # Çıkıştan girişe/gizliye veya gizliden girişe bağlantı yok (ileri beslemeli varsayımı)
            # TODO: Döngüsel bağlantılara izin vermek için bu kontrol kaldırılabilir/değiştirilebilir.
            start_node = self.node_genes[start_node_id]
            end_node = self.node_genes[end_node_id]
            if start_node.type == NodeType.OUTPUT or end_node.type == NodeType.INPUT: continue
            if start_node.type == NodeType.HIDDEN and end_node.type == NodeType.HIDDEN: pass # Gizliden gizliye olabilir
            if start_node.type == NodeType.INPUT and end_node.type == NodeType.OUTPUT: pass # Girişten çıkışa olabilir
            if start_node.type == NodeType.INPUT and end_node.type == NodeType.HIDDEN: pass # Girişten gizliye olabilir
            if start_node.type == NodeType.HIDDEN and end_node.type == NodeType.OUTPUT: pass # Gizliden çıkışa olabilir


            # Bağlantı zaten var mı kontrol et
            connection_exists = False
            for conn in self.connection_genes.values():
                if (conn.in_node_id == start_node_id and conn.out_node_id == end_node_id) or \
                   (conn.in_node_id == end_node_id and conn.out_node_id == start_node_id): # Ters yönlü de kontrol et
                    connection_exists = True
                    break

            if not connection_exists:
                # Yeni bağlantıyı ekle
                new_weight = np.float32(np.random.randn() * 0.1) # Küçük rastgele ağırlık
                new_innov_num = innovation_tracker.get_new_innovation_number()
                new_conn = ConnectionGene(start_node_id, end_node_id, new_weight, True, new_innov_num)
                self.add_connection_gene(new_conn)
                logging.debug(f"Genome {self.id}: Added new connection {new_conn}")
                return # Başarıyla eklendi, çık

        # logging.warning(f"Genome {self.id}: Could not find nodes to add a new connection after {max_attempts} attempts.")


    def mutate_add_node(self, innovation_tracker: InnovationTracker):
        """ Mevcut bir bağlantıyı bölerek araya yeni bir gizli düğüm ekler. """
        enabled_connections = [c for c in self.connection_genes.values() if c.enabled]
        if not enabled_connections: return

        # Bölünecek bağlantıyı seç
        conn_to_split = random.choice(enabled_connections)
        conn_to_split.enabled = False # Eski bağlantıyı devre dışı bırak

        # Yeni düğümü oluştur
        new_node_id = innovation_tracker.get_new_node_id()
        # Yeni düğümün bias'ı genellikle 0 başlatılır, aktivasyonu miras alabilir veya varsayılan olabilir
        new_node = NodeGene(new_node_id, NodeType.HIDDEN, bias=0.0, activation='sigmoid') # Veya conn_to_split'in çıktısındaki aktivasyon?
        self.add_node_gene(new_node)

        # Yeni bağlantıları oluştur
        # Giriş -> Yeni Düğüm (ağırlık=1)
        innov1 = innovation_tracker.get_new_innovation_number()
        conn1 = ConnectionGene(conn_to_split.in_node_id, new_node_id, 1.0, True, innov1)
        # Yeni Düğüm -> Çıkış (ağırlık = eski bağlantının ağırlığı)
        innov2 = innovation_tracker.get_new_innovation_number()
        conn2 = ConnectionGene(new_node_id, conn_to_split.out_node_id, conn_to_split.weight, True, innov2)

        self.add_connection_gene(conn1)
        self.add_connection_gene(conn2)
        logging.debug(f"Genome {self.id}: Added new node {new_node_id} splitting connection {conn_to_split.innovation_number}. New conns: {innov1}, {innov2}")

    def get_phenotype_model(self, device: torch.device) -> Optional['FeedForwardNetwork']:
        """ Genomdan basit bir ileri beslemeli PyTorch modeli (fenotip) oluşturur. """
        try:
            return FeedForwardNetwork(self, device)
        except Exception as e:
            logging.error(f"Genome {self.id}: Failed to create phenotype model: {e}", exc_info=True)
            return None

    def copy(self) -> 'Genome':
        new_genome = Genome(self.id) # ID'yi koruyalım mı? Klon için belki yeni ID? Şimdilik koruyalım.
        new_genome.node_genes = {nid: node.copy() for nid, node in self.node_genes.items()}
        new_genome.connection_genes = {innov: conn.copy() for innov, conn in self.connection_genes.items()}
        new_genome.fitness = self.fitness
        return new_genome

    def __repr__(self) -> str:
        return f"Genome(id={self.id}, nodes={len(self.node_genes)}, conns={len(self.connection_genes)}, fitness={self.fitness})"

    def to_dict(self) -> Dict[str, Any]:
         # Checkpoint için genomu serileştir
         return {
              'id': self.id,
              'nodes': {nid: node.to_dict() for nid, node in self.node_genes.items()},
              'connections': {innov: conn.to_dict() for innov, conn in self.connection_genes.items()},
              'fitness': self.fitness
         }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
         genome = cls(genome_id=data['id'])
         genome.node_genes = {int(nid): NodeGene.from_dict(ndata) for nid, ndata in data['nodes'].items()}
         genome.connection_genes = {int(innov): ConnectionGene.from_dict(cdata) for innov, cdata in data['connections'].items()}
         genome.fitness = data.get('fitness')
         return genome

# === v6: Fenotip Modeli (Basit İleri Beslemeli) ===

def _sigmoid(x): return 1 / (1 + np.exp(-x * 4.9)) # NEAT'teki gibi eğimi ayarlanmış sigmoid
def _relu(x): return max(0, x)
def _tanh(x): return math.tanh(x)
def _identity(x): return x

ACTIVATION_FUNCTIONS = {
    'sigmoid': _sigmoid,
    'relu': _relu,
    'tanh': _tanh,
    'identity': _identity,
}

class FeedForwardNetwork(nn.Module):
    """ Genomdan oluşturulan basit ileri beslemeli ağ.
        Not: Bu implementasyon oldukça basittir ve döngüsel bağlantıları veya
        karmaşık katman yapılarını desteklemez. Aktivasyonu adım adım yapar.
    """
    def __init__(self, genome: Genome, device: torch.device):
        super().__init__()
        self.genome = genome # Referansı sakla? Veya sadece gerekli bilgiyi kopyala? Kopyalamak daha güvenli.
        self.input_node_ids = sorted([n.id for n in genome.node_genes.values() if n.type == NodeType.INPUT])
        self.output_node_ids = sorted([n.id for n in genome.node_genes.values() if n.type == NodeType.OUTPUT])
        self.all_node_ids = sorted(genome.node_genes.keys())
        self.device = device # Bu modelin hangi cihazda çalışacağı

        # Bağlantıları ve düğüm bilgilerini hazırla
        self.connections = {} # out_node_id -> list of (in_node_id, weight)
        self.node_eval_order = self._determine_eval_order() # Düğümleri hangi sırada hesaplayacağımızı belirle
        self.node_details = {nid: {'bias': gene.bias, 'activation': gene.activation}
                             for nid, gene in genome.node_genes.items()}

        for conn in genome.connection_genes.values():
            if conn.enabled:
                out_id = conn.out_node_id
                if out_id not in self.connections:
                    self.connections[out_id] = []
                self.connections[out_id].append((conn.in_node_id, conn.weight))

        # Ağırlıkları veya katmanları PyTorch parametresi olarak kaydetmek gerekir mi?
        # Bu dinamik yapıda zor. Şimdilik `forward` içinde doğrudan kullanacağız.
        # Bu, gradyan tabanlı eğitim için uygun DEĞİLDİR. Sadece inferans için.
        self.model_name = f"phenotype_genome_{genome.id}"
        self.to(device) # Cihaza taşıma denemesi (içinde parametre olmasa da)

    def _determine_eval_order(self) -> List[int]:
        """ Düğümlerin hesaplama sırasını belirler (basit topolojik sıralama).
            Bu implementasyon sadece ileri beslemeli ağlar için çalışır.
        """
        # Basit yaklaşım: Giriş -> Gizli -> Çıkış
        # Daha sağlam: Gelen bağlantılarına göre katmanları belirle
        layers: List[Set[int]] = [set(self.input_node_ids)]
        processed_nodes = set(self.input_node_ids)

        while True:
            next_layer_nodes = set()
            nodes_in_current_layers = set().union(*layers)

            for node_id in self.genome.node_genes:
                if node_id not in processed_nodes and node_id not in self.input_node_ids:
                    node_gene = self.genome.node_genes[node_id]
                    # Bu düğüme gelen tüm etkin bağlantıların başlangıç düğümleri
                    # önceki katmanlarda işlendi mi?
                    incoming_enabled_inputs = set()
                    for conn in self.genome.connection_genes.values():
                         if conn.enabled and conn.out_node_id == node_id:
                              incoming_enabled_inputs.add(conn.in_node_id)

                    if incoming_enabled_inputs.issubset(nodes_in_current_layers):
                        next_layer_nodes.add(node_id)

            if not next_layer_nodes:
                break # Yeni eklenecek düğüm kalmadı

            layers.append(next_layer_nodes)
            processed_nodes.update(next_layer_nodes)

            # Güvenlik: Sonsuz döngü kontrolü (eğer döngüsel bağlantı varsa olabilir)
            if len(processed_nodes) > len(self.genome.node_genes) * 2:
                 raise RuntimeError(f"Could not determine evaluation order for Genome {self.genome.id}. Possible cycle detected or logic error.")


        # Katmanları düzleştirerek sıralı listeyi oluştur
        eval_order = []
        for layer in layers:
            # Katman içindeki sıranın önemi var mı? Şimdilik ID'ye göre sıralayalım.
            eval_order.extend(sorted(list(layer)))

        # Çıktı düğümlerinin sonda olduğundan emin ol (veya sıralamada yer aldığından)
        ordered_set = set(eval_order)
        if not set(self.output_node_ids).issubset(ordered_set):
             # Bu durum, çıkış düğümlerine hiçbir bağlantı yoksa veya
             # döngüsel bir yapı varsa oluşabilir.
             logging.warning(f"Genome {self.genome.id}: Not all output nodes are reachable in feed-forward pass. Eval order: {eval_order}")
             # Eksik çıkış düğümlerini sona ekleyebiliriz, ancak değerleri 0 olabilir.
             for out_id in self.output_node_ids:
                  if out_id not in ordered_set: eval_order.append(out_id)


        # Giriş düğümlerini hesaplama sırasından çıkaralım, onlar doğrudan input alır.
        eval_order_no_inputs = [node_id for node_id in eval_order if node_id not in self.input_node_ids]

        # print(f"DEBUG Genome {self.genome.id} Eval order: {eval_order_no_inputs}")
        return eval_order_no_inputs


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Ağı ileri beslemeli olarak çalıştırır. """
        if x.shape[1] != len(self.input_node_ids):
            raise ValueError(f"Input tensor shape mismatch: expected {len(self.input_node_ids)} features, got {x.shape[1]}")

        # Düğüm aktivasyonlarını saklamak için dict
        # NumPy dizisi üzerinde çalışmak daha kolay olabilir, sonra Tensor'a çeviririz.
        # Batch processing için: (batch_size, num_nodes) boyutlu array.
        batch_size = x.shape[0]
        # Tüm düğümler için yer ayıralım, ID'leri indeks olarak kullanalım (maksimum ID'ye göre boyutlandır)
        max_node_id = max(self.all_node_ids) if self.all_node_ids else -1
        node_values = np.zeros((batch_size, max_node_id + 1), dtype=np.float32)

        # Giriş değerlerini ata (NumPy'a çevirerek)
        input_data_np = x.cpu().numpy()
        for i, node_id in enumerate(self.input_node_ids):
             if node_id <= max_node_id:
                 node_values[:, node_id] = input_data_np[:, i]

        # Hesaplama sırasına göre düğümleri aktive et
        for node_id in self.node_eval_order:
            if node_id not in self.node_details: continue # Sıralamada olmayan düğüm?

            node_info = self.node_details[node_id]
            activation_func = ACTIVATION_FUNCTIONS.get(node_info['activation'], _identity)
            node_bias = node_info['bias']

            # Gelen bağlantılardan gelen toplam girdiyi hesapla
            node_input_sum = np.full((batch_size,), node_bias, dtype=np.float32) # Bias ile başla
            if node_id in self.connections:
                for in_node_id, weight in self.connections[node_id]:
                     # Önceki düğümün değerini al (hesaplanmış olmalı)
                     if in_node_id <= max_node_id:
                         incoming_values = node_values[:, in_node_id]
                         node_input_sum += incoming_values * weight
                     #else: logging.warning(f"Node {in_node_id} not found in node_values array during forward pass.")


            # Aktivasyon fonksiyonunu uygula (vektörel olarak)
            # node_output = activation_func(node_input_sum) # Bu tekil değer için çalışır
            # NumPy ile vektörize etmek lazım
            if node_info['activation'] == 'sigmoid': node_output = _sigmoid(node_input_sum)
            elif node_info['activation'] == 'relu': node_output = np.maximum(0, node_input_sum)
            elif node_info['activation'] == 'tanh': node_output = np.tanh(node_input_sum)
            else: node_output = node_input_sum # identity

            if node_id <= max_node_id:
                node_values[:, node_id] = node_output
            #else: logging.warning(f"Node {node_id} not found in node_values array when storing output.")


        # Çıkış düğümlerinin değerlerini topla ve Tensor'a çevir
        output_values = np.stack([node_values[:, out_id] for out_id in self.output_node_ids], axis=1)

        return torch.from_numpy(output_values).float().to(self.device)


# --- Adaptif Mutasyon Gücü Yardımcı Fonksiyonu ---
# Bu, evrim döngüsü içinde kullanılacak global değişkenler yerine
# durumu takip etmek için daha iyi bir yol olabilir (örn: bir sınıf).
# Şimdilik basit tutalım ve döngü içinde yönetelim.
# Bu fonksiyon, mutate çağrısında kullanılacak mevcut gücü döndürür (yer tutucu).
def get_current_adaptive_strength(args: argparse.Namespace) -> float:
     # Bu fonksiyon aslında evolve_population içinde hesaplanan
     # `current_mutation_strength` değerini almalı.
     # Direkt mutate içinde hesaplamak yerine evolve'da hesaplayıp geçmek daha iyi.
     # Bu yüzden bu fonksiyonu kaldırıp evolve içinde direkt kullanalım.
     pass


# --- v6: Evrim Döngüsü (Genom Tabanlı) ---
def evolve_population_pytorch_v6(
    population: List[Genome], # Artık Genom listesi
    X_train_np: np.ndarray, y_train_np: np.ndarray,
    start_generation: int, total_generations: int,
    args: argparse.Namespace,
    output_dir: str, device: torch.device,
    innovation_tracker: InnovationTracker, # Küresel tracker'ı al
    wandb_run: Optional[Any]
) -> Tuple[Optional[Genome], List[float], List[float]]:
    """ PyTorch v6 Genom tabanlı evrimsel süreci çalıştırır. """

    best_fitness_history = []
    avg_fitness_history = []
    best_genome_overall: Optional[Genome] = None
    best_fitness_overall = -np.inf

    current_mutation_strength = args.mutation_strength
    stagnation_counter = 0
    pop_size = len(population)
    fitness_params = {'complexity_penalty': args.complexity_penalty}

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) if args.num_workers > 0 else None
    if executor: logging.info(f"v6: Using ProcessPoolExecutor with {args.num_workers} workers.")

    try:
        for gen in range(start_generation, total_generations):
            generation_start_time = time.time()

            # 1. Fitness Değerlendirme (Paralel/Seri - Genomlarla)
            fitness_scores = [-np.inf] * pop_size
            genomes_to_eval = population # Doğrudan genom listesi

            try:
                # Fenotip oluşturma potansiyel olarak hata verebilir, bunu da yakala
                phenotype_creation_errors = [False] * pop_size

                if executor and args.num_workers > 0:
                    futures_map = {}
                    for i, genome in enumerate(genomes_to_eval):
                         # Genomun serileştirilebilir kopyasını gönderelim mi?
                         # Ya da sadece dict temsilini? Dict daha güvenli olabilir.
                         genome_dict = genome.to_dict()
                         futures_map[executor.submit(_calculate_fitness_worker_v6,
                                                     genome_dict, X_train_np, y_train_np,
                                                     str(device), fitness_params)] = i

                    for future in concurrent.futures.as_completed(futures_map):
                        original_index = futures_map[future]
                        try:
                            result = future.result()
                            # Eğer worker None döndürdüyse (örn: fenotip hatası)
                            if result is None:
                                fitness_scores[original_index] = -np.inf
                                phenotype_creation_errors[original_index] = True
                            else:
                                fitness_scores[original_index] = result
                                genomes_to_eval[original_index].fitness = result # Genomun fitness'ını güncelle
                        except Exception as exc:
                            logging.error(f'Genome {genomes_to_eval[original_index].id} fitness calc generated exception: {exc}')
                            fitness_scores[original_index] = -np.inf
                            phenotype_creation_errors[original_index] = True
                else: # Seri
                    logging.debug("Calculating fitness sequentially (v6)...")
                    temp_device = torch.device("cpu")
                    for i, genome in enumerate(genomes_to_eval):
                        try:
                             fitness = calculate_fitness_pytorch_v6(genome, X_train_np, y_train_np, temp_device, fitness_params)
                             if fitness is None: # Fenotip hatası
                                  fitness_scores[i] = -np.inf
                                  phenotype_creation_errors[i] = True
                             else:
                                  fitness_scores[i] = fitness
                                  genome.fitness = fitness
                        except Exception as e:
                             logging.error(f"Error calculating fitness for genome {genome.id} sequentially: {e}")
                             fitness_scores[i] = -np.inf
                             phenotype_creation_errors[i] = True

                num_phenotype_errors = sum(phenotype_creation_errors)
                if num_phenotype_errors > 0:
                    logging.warning(f"Generation {gen+1}: {num_phenotype_errors}/{pop_size} individuals failed phenotype creation or fitness calculation.")

            except Exception as e:
                logging.critical(f"Error during fitness evaluation distribution/collection in Gen {gen+1}: {e}", exc_info=True)
                raise

            # 2. İstatistikler ve En İyiyi Takip
            valid_indices = [i for i, score in enumerate(fitness_scores) if np.isfinite(score)]
            if not valid_indices:
                logging.error(f"Generation {gen+1}: No individuals with finite fitness scores! Evolution stopped.")
                raise RuntimeError(f"Evolution stopped at gen {gen+1} due to lack of valid individuals.")

            current_best_idx_local = np.argmax([fitness_scores[i] for i in valid_indices])
            current_best_idx_global = valid_indices[current_best_idx_local]
            current_best_fitness = fitness_scores[current_best_idx_global]
            current_best_genome = population[current_best_idx_global]

            finite_scores = [fitness_scores[i] for i in valid_indices]
            avg_fitness = np.mean(finite_scores)

            best_fitness_history.append(current_best_fitness)
            avg_fitness_history.append(avg_fitness)

            new_best_found = False
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                new_best_found = True
                try:
                    best_genome_overall = current_best_genome.copy() # En iyi genomu kopyala
                    logging.info(f"Generation {gen+1}: *** New overall best fitness: {best_fitness_overall:.6f} (Genome ID: {best_genome_overall.id}) ***")
                    # En iyi genomun detaylarını logla (opsiyonel)
                    logging.info(f"Best Genome Stats: Nodes={len(best_genome_overall.node_genes)}, Conns={len(best_genome_overall.connection_genes)}")
                except Exception as e:
                    logging.error(f"Could not copy new best genome {current_best_genome.id}: {e}", exc_info=True)
                    best_genome_overall = None

            generation_time = time.time() - generation_start_time
            logging.info(f"Generation {gen+1}/{total_generations} | Best Fit: {current_best_fitness:.6f} | Avg Fit: {avg_fitness:.6f} | Mut Str: {current_mutation_strength:.4f} | Valid: {len(valid_indices)}/{pop_size} | Time: {generation_time:.2f}s")

            # W&B Loglama
            if wandb_run:
                log_data = {"generation": gen + 1, "best_fitness": current_best_fitness, "average_fitness": avg_fitness,
                            "mutation_strength": current_mutation_strength, "generation_time_sec": generation_time,
                            "num_valid_individuals": len(valid_indices), "num_phenotype_errors": num_phenotype_errors}
                if best_genome_overall:
                     log_data["best_genome_nodes"] = len(best_genome_overall.node_genes)
                     log_data["best_genome_conns"] = len(best_genome_overall.connection_genes)
                try: wandb_run.log(log_data, step=gen + 1)
                except Exception as e: logging.warning(f"Failed to log metrics to W&B: {e}")

            # Adaptif Mutasyon Gücü Güncelleme
            if args.adapt_mutation:
                if new_best_found: stagnation_counter = 0; current_mutation_strength = max(args.min_mut_strength, current_mutation_strength * args.mut_strength_decay)
                else:
                    stagnation_counter += 1
                    if stagnation_counter >= args.stagnation_limit:
                        current_mutation_strength = min(args.max_mut_strength, current_mutation_strength * args.mut_strength_increase)
                        logging.info(f"Stagnation detected. Increasing mutation strength to {current_mutation_strength:.4f}")
                        stagnation_counter = 0

            # 3. Yeni Popülasyon Oluşturma
            new_population = []

            # 3a. Elitizm (Genomları kopyala)
            if args.elitism_count > 0 and len(population) >= args.elitism_count:
                try:
                    sorted_valid_indices = sorted(valid_indices, key=lambda i: fitness_scores[i], reverse=True)
                    elite_indices = sorted_valid_indices[:args.elitism_count]
                    for idx in elite_indices:
                        elite_clone = population[idx].copy() # Genomu kopyala
                        # elite_clone.id = next_genome_id() # ID'yi değiştir? Opsiyonel.
                        new_population.append(elite_clone)
                    logging.debug(f"Added {len(new_population)} elite genomes.")
                except Exception as e: logging.error(f"Error during elitism: {e}", exc_info=True)

            # 3b. Kalanları Üretme (Şimdilik sadece mutasyonla)
            num_to_generate = pop_size - len(new_population)
            generated_count = 0
            reproduction_attempts = 0
            max_reproduction_attempts = num_to_generate * 5

            while generated_count < num_to_generate and reproduction_attempts < max_reproduction_attempts:
                reproduction_attempts += 1
                try:
                    # Ebeveyn seç (fitness'a göre)
                    parent_genome = tournament_selection(population, fitness_scores, args.tournament_size)
                    # Ebeveyni kopyala ve mutasyona uğrat
                    child_genome = parent_genome.copy()
                    # Mutasyon oranlarını ve adaptif gücü içeren args'ı mutate'e verelim
                    # Adaptif gücü mutate fonksiyonu içinde kullanmak yerine burada geçirelim.
                    args.current_mutation_strength = current_mutation_strength # Geçici olarak ekle
                    child_genome.mutate(innovation_tracker, args)
                    del args.current_mutation_strength # Geri sil

                    new_population.append(child_genome)
                    generated_count += 1

                except Exception as e:
                    logging.error(f"Error during selection/reproduction cycle (attempt {reproduction_attempts}): {e}", exc_info=True)

            # Popülasyonu tamamla (gerekirse rastgele *minimal* genomlarla)
            if generated_count < num_to_generate:
                logging.warning(f"Reproduction cycle failed to generate enough individuals. Adding {num_to_generate - generated_count} new minimal genomes.")
                # Başlangıç genomu oluşturma fonksiyonuna ihtiyacımız var
                input_size = args.seq_length # Varsayım, config'den gelmeli
                output_size = args.seq_length # Varsayım
                for _ in range(num_to_generate - generated_count):
                    try:
                         new_minimal_genome = create_initial_genome(input_size, output_size, innovation_tracker)
                         new_population.append(new_minimal_genome)
                    except Exception as e:
                         logging.error(f"Failed to create minimal genome to fill population: {e}")


            population = new_population[:pop_size]

            # 4. Checkpoint Alma (Genomları ve Tracker Durumunu Kaydet)
            if args.checkpoint_interval > 0 and (gen + 1) % args.checkpoint_interval == 0:
                 try:
                     rnd_state = random.getstate()
                     np_rnd_state = np.random.get_state()
                     torch_rnd_state = torch.get_rng_state().cpu()
                     tracker_state = innovation_tracker.get_state()
                     wandb_id = wandb_run.id if wandb_run else None
                     save_checkpoint_pytorch_v6(output_dir, gen + 1, population, tracker_state,
                                                rnd_state, np_rnd_state, torch_rnd_state, wandb_id)
                 except Exception as e:
                     logging.error(f"Failed to execute checkpoint saving for generation {gen+1}: {e}", exc_info=True)

    finally: # Executor'ı kapat
        if executor:
            logging.info("Shutting down ProcessPoolExecutor (v6)...")
            executor.shutdown(wait=True)
            logging.info("Executor shut down.")


    # Evrim Sonu (En iyi genomu seç)
    if best_genome_overall is None and population:
        logging.warning("Evolution finished, tracking failed. Selecting best from final pop.")
        # Son fitness'ları kullanarak en iyiyi seç (tekrar hesaplamaya gerek yok)
        final_valid_indices = [i for i, p in enumerate(population) if p.fitness is not None and np.isfinite(p.fitness)]
        if final_valid_indices:
            best_idx_final = max(final_valid_indices, key=lambda i: population[i].fitness)
            best_genome_overall = population[best_idx_final].copy()
            best_fitness_overall = best_genome_overall.fitness
            logging.info(f"Selected best genome from final population: ID {best_genome_overall.id} with fitness {best_fitness_overall:.6f}")
        else:
            logging.error("Evolution finished. No valid finite fitness scores in the final population.")
            return None, best_fitness_history, avg_fitness_history
    elif not population:
         logging.error("Evolution finished with an empty population!")
         return None, best_fitness_history, avg_fitness_history
    else: # best_genome_overall zaten bulundu
         logging.info(f"Evolution finished. Best fitness achieved: {best_fitness_overall:.6f} by genome {best_genome_overall.id}")

    return best_genome_overall, best_fitness_history, avg_fitness_history


# --- v6: Fitness Hesaplama (Paralel İşçi - Genom ile) ---
def _calculate_fitness_worker_v6(
    genome_dict: Dict[str, Any], # Artık genom dict'i alıyor
    X_np: np.ndarray, y_np: np.ndarray,
    device_str: str, fitness_params: Dict
) -> Optional[float]: # Fenotip hatasında None dönebilir
    """ Bir genomun fitness'ını hesaplayan işçi fonksiyonu (v6). """
    try:
        # 1. Genomu dict'ten oluştur
        genome = Genome.from_dict(genome_dict)

        # 2. Fenotip modelini oluştur
        device = torch.device(device_str)
        phenotype_model = genome.get_phenotype_model(device)

        # Eğer model oluşturulamadıysa (örn: hata veya döngü)
        if phenotype_model is None:
             print(f"[Worker Error] Failed to create phenotype for Genome {genome.id}", file=sys.stderr)
             return None # Hata kodu olarak None

        phenotype_model.eval()

        # 3. Veriyi Tensör'e çevir ve cihaza taşı
        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)

        # 4. Fitness Hesaplama
        complexity_penalty_weight = fitness_params.get('complexity_penalty', 0.0)

        with torch.no_grad():
            y_pred = phenotype_model(X)
            mse_val = torch.mean((y_pred - y)**2).item()

        # ... (geri kalan fitness hesaplama mantığı _calculate_fitness_worker ile aynı) ...
        if not np.isfinite(mse_val): return -np.inf
        fitness_score = 1.0 / (mse_val + 1e-9)
        if complexity_penalty_weight > 0:
            # Karmaşıklığı genomdan alabiliriz (parametre sayısını fenotipten almak yerine)
            # Fenotipteki etkin parametre sayısını saymak daha doğru olabilir.
            # Şimdilik fenotip modelinden alalım.
            try:
                # Fenotip basit olduğu için parametreleri yok, genomdan alalım.
                # TODO: Fenotip modeline parametre sayma eklemek veya genomdan almak.
                # Basitçe bağlantı sayısı + düğüm sayısı (bias) diyelim:
                num_conns = len([c for c in genome.connection_genes.values() if c.enabled])
                num_nodes = len(genome.node_genes) # Bias olan düğümler?
                complexity = num_conns + num_nodes # Çok kaba bir ölçüm!
                fitness_score -= complexity_penalty_weight * complexity
            except Exception as ce:
                 print(f"[Worker Warning] Could not calculate complexity for genome {genome.id}: {ce}", file=sys.stderr)


        if not np.isfinite(fitness_score): return -np.inf
        return float(fitness_score)

    except Exception as e:
        print(f"[Worker Error] Exception in fitness worker for genome {genome_dict.get('id','N/A')}: {e}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr) # Detaylı hata için
        return None # Hata durumunda None dönelim

# --- v6: Fitness Hesaplama (Seri - Genom ile) ---
def calculate_fitness_pytorch_v6(
    genome: Genome,
    X_np: np.ndarray, y_np: np.ndarray,
    device: torch.device,
    fitness_params: Dict
) -> Optional[float]: # Fenotip hatasında None dönebilir
    """ Bir genomun fitness değerini hesaplar (Seri kullanım v6). """
    phenotype_model = genome.get_phenotype_model(device)
    if phenotype_model is None:
         logging.warning(f"Genome {genome.id}: Failed phenotype creation (Serial).")
         return None

    phenotype_model.eval()
    try:
        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)
    except Exception as e:
         logging.error(f"Genome {genome.id}: Error converting data in calculate_fitness_v6: {e}")
         return -np.inf # Veri hatası

    complexity_penalty_weight = fitness_params.get('complexity_penalty', 0.0)
    try:
        with torch.no_grad():
            y_pred = phenotype_model(X)
            mse_val = torch.mean((y_pred - y)**2).item()
        if not np.isfinite(mse_val): return -np.inf
        fitness_score = 1.0 / (mse_val + 1e-9)
        if complexity_penalty_weight > 0:
            # complexity = phenotype_model.get_num_params() # Eğer fenotip sayabilseydi
            num_conns = len([c for c in genome.connection_genes.values() if c.enabled])
            num_nodes = len(genome.node_genes)
            complexity = num_conns + num_nodes
            fitness_score -= complexity_penalty_weight * complexity
        if not np.isfinite(fitness_score): return -np.inf
        return float(fitness_score)
    except Exception as e:
        logging.error(f"Error during serial fitness calculation for genome {genome.id}: {e}", exc_info=True)
        return -np.inf


# --- v6: Checkpointing (Genomları ve Tracker'ı Kaydet) ---
def save_checkpoint_pytorch_v6(output_dir: str, generation: int, population: List[Genome],
                               tracker_state: Dict[str, int], # Tracker durumu eklendi
                               rnd_state: Any, np_rnd_state: Any, torch_rnd_state: Any,
                               wandb_run_id: Optional[str] = None):
    """ Evrim durumunu (PyTorch v6 - Genomlar) kaydeder. """
    checkpoint_dir = os.path.join(output_dir, "checkpoints_pytorch_v6")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"evo_gen_{generation}.pt")
    logging.info(f"Saving v6 checkpoint for generation {generation} to {checkpoint_file}...")

    population_state_dicts = []
    for genome in population:
        try: population_state_dicts.append(genome.to_dict())
        except Exception as e: logging.error(f"Could not serialize genome {genome.id} for checkpoint: {e}")

    state = {
        "version": "v6_base", # Sürüm detayı
        "generation": generation,
        "population_state": population_state_dicts, # Genom dict listesi
        "innovation_tracker_state": tracker_state, # Tracker durumu
        "random_state": rnd_state,
        "numpy_random_state": np_rnd_state,
        "torch_random_state": torch_rnd_state, # CPU state olmalı
        "wandb_run_id": wandb_run_id,
        "timestamp": datetime.now().isoformat()
    }
    try:
        torch.save(state, checkpoint_file)
        logging.info(f"v6 Checkpoint saved successfully for generation {generation}.")
    except Exception as e:
        logging.error(f"Failed to save v6 checkpoint using torch.save for generation {generation}: {e}", exc_info=True)

# --- v6: Checkpoint Yükleme (Genomları ve Tracker'ı Yükle) ---
def load_checkpoint_pytorch_v6(checkpoint_path: str) -> Optional[Dict]:
    """ Kaydedilmiş PyTorch v6 evrim durumunu (Genomlar) yükler. """
    if not os.path.exists(checkpoint_path): logging.error(f"Checkpoint file not found: {checkpoint_path}"); return None
    logging.info(f"Loading v6 checkpoint from {checkpoint_path}...")
    try:
        # Checkpoint'i CPU'ya yükle
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # if "v6" not in checkpoint.get("version", ""): # Daha spesifik kontrol
        #      logging.warning(f"Loading non-v6 checkpoint ({checkpoint.get('version', 'Unknown')}).")

        population = []
        for genome_dict in checkpoint["population_state"]:
            try: population.append(Genome.from_dict(genome_dict))
            except Exception as e: logging.error(f"Failed to load genome state from checkpoint for genome {genome_dict.get('id', 'UNKNOWN')}: {e}", exc_info=True)

        if not population: logging.error("Failed to load any genome from the checkpoint."); return None

        checkpoint["population"] = population
        logging.info(f"Checkpoint loaded successfully. Resuming from generation {checkpoint['generation'] + 1}.")
        # Tracker durumu ve W&B ID'sini de döndür
        checkpoint["innovation_tracker_state"] = checkpoint.get("innovation_tracker_state")
        checkpoint["wandb_run_id"] = checkpoint.get("wandb_run_id")
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load v6 checkpoint from {checkpoint_path}: {e}", exc_info=True)
        return None

# --- v6: Başlangıç Genomu Oluşturma ---
def create_initial_genome(input_size: int, output_size: int, innovation_tracker: InnovationTracker) -> Genome:
    """ Minimal başlangıç genomu oluşturur (girişler doğrudan çıkışlara bağlı). """
    genome = Genome()
    node_id_map = {} # Geçici ID'den gerçek ID'ye

    # Giriş düğümleri
    input_ids = []
    for i in range(input_size):
         node_id = innovation_tracker.get_new_node_id()
         node_id_map[f'in_{i}'] = node_id
         input_ids.append(node_id)
         genome.add_node_gene(NodeGene(node_id, NodeType.INPUT, bias=0.0, activation='identity')) # Girişler bias'sız ve identity

    # Çıkış düğümleri
    output_ids = []
    for i in range(output_size):
         node_id = innovation_tracker.get_new_node_id()
         node_id_map[f'out_{i}'] = node_id
         output_ids.append(node_id)
         # Çıkış aktivasyonu probleme göre değişir, varsayılan sigmoid olsun
         genome.add_node_gene(NodeGene(node_id, NodeType.OUTPUT, bias=np.float32(np.random.randn()*0.1), activation='sigmoid'))

    # İsteğe bağlı: Başlangıçta girişleri çıkışlara bağla
    for i_id in input_ids:
         for o_id in output_ids:
              weight = np.float32(np.random.randn() * 0.1) # Küçük rastgele ağırlık
              innov_num = innovation_tracker.get_new_innovation_number()
              genome.add_connection_gene(ConnectionGene(i_id, o_id, weight, True, innov_num))

    logging.debug(f"Created initial minimal genome {genome.id} with {len(input_ids)} inputs, {len(output_ids)} outputs.")
    return genome


# --- Grafik/Değerlendirme/Eğitim (v5'ten benzer, ancak Genom/Fenotip ile çalışmalı) ---
# plot_fitness_history v5'teki gibi kalabilir.

def evaluate_model_pytorch_v6(
    genome: Genome, # Artık genom alıyor
    X_test_np: np.ndarray, y_test_np: np.ndarray,
    batch_size: int, device: torch.device
) -> Dict[str, float]:
    """ En iyi genomun fenotipini test verisi üzerinde değerlendirir (v6). """
    if genome is None: logging.error("Cannot evaluate a None genome."); return {"test_mse": np.inf, "avg_kendall_tau": 0.0}
    logging.info(f"Evaluating final genome {genome.id} on test data (PyTorch v6)...")
    # Fenotipi oluştur
    model = genome.get_phenotype_model(device)
    if model is None:
        logging.error(f"Failed to create phenotype for final genome {genome.id}. Evaluation skipped.")
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}

    # Geri kalanı evaluate_model_pytorch (v5) ile aynı, modeli kullanır.
    model.eval() # Zaten eval modunda olmalı ama yine de set edelim.
    # ... (v5'teki DataLoader ve hesaplama kısmı buraya gelir) ...
    try:
        test_dataset = TensorDataset(torch.from_numpy(X_test_np).float(), torch.from_numpy(y_test_np).float())
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        all_preds, all_targets = [], []; total_mse, num_batches = 0.0, 0
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
        all_preds_np = np.concatenate(all_preds, axis=0); all_targets_np = np.concatenate(all_targets, axis=0)
        sample_size = min(500, all_targets_np.shape[0]); taus = []
        if sample_size > 0:
            indices = np.random.choice(all_targets_np.shape[0], sample_size, replace=False)
            for i in indices:
                try: tau, _ = kendalltau(all_targets_np[i], all_preds_np[i]);
                if not np.isnan(tau): taus.append(tau)
                except ValueError: pass
        avg_kendall_tau = np.mean(taus) if taus else 0.0
        logging.info(f"Average Kendall's Tau (on {sample_size} samples): {avg_kendall_tau:.4f}")
        return {"test_mse": float(avg_mse), "avg_kendall_tau": float(avg_kendall_tau)}
    except Exception as e:
        logging.error(f"Error during final genome evaluation: {e}", exc_info=True)
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}


# Final Training: v6'da genomdan oluşturulan fenotip genellikle doğrudan
# gradyan tabanlı eğitim için uygun DEĞİLDİR (parametreleri yok/dinamik).
# Bu adım ya atlanmalı, ya genomdan daha standart bir model (örn: MLP)
# oluşturulup o eğitilmeli, ya da gradyanları genomdaki ağırlıklara/biaslara
# geri yayacak karmaşık bir yöntem geliştirilmelidir.
# Şimdilik en basit yol, bu adımı atlamak veya sadece en iyi genomu kaydetmektir.
# Biz atlayalım ve sadece değerlendirme yapalım.

# --- Ana İş Akışı (PyTorch v6) ---
def run_pipeline_pytorch_v6(args: argparse.Namespace):
    """ Genom tabanlı v6 ana iş akışı. """
    wandb_run = None; output_dir = None
    try:
        device = setup_device(args.device)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"evorun_pt_v6_{timestamp}_gen{args.generations}_pop{args.pop_size}"
        output_dir = args.resume_from if args.resume_from else os.path.join(args.output_base_dir, run_name)
        resume_run = bool(args.resume_from); resumed_wandb_id = None; tracker_state = None

        if resume_run:
            # ... (v5'teki resume klasör kontrolü) ...
            run_name = os.path.basename(output_dir)
        else:
            # ... (v5'teki output_dir oluşturma) ...
            pass # Önceki kodu buraya yapıştırın

        setup_logging(output_dir) # v6 log başlığı ile
        logging.info(f"========== Starting/Resuming EvoNet v6 PyTorch Pipeline: {run_name} ==========")
        # ... (v5'teki output_dir, device loglama) ...

        # Checkpoint Yükleme (v6)
        start_generation = 0; population: List[Genome] = []; initial_state_loaded = False
        latest_checkpoint_path = find_latest_checkpoint_pytorch(output_dir) if resume_run else None # v6 klasörünü arar

        if latest_checkpoint_path:
            loaded_state = load_checkpoint_pytorch_v6(latest_checkpoint_path) # v6 yükleyici
            if loaded_state:
                start_generation = loaded_state['generation']
                population = loaded_state['population'] # Genom listesi
                tracker_state = loaded_state.get("innovation_tracker_state")
                resumed_wandb_id = loaded_state.get("wandb_run_id")
                try: # Random state yükleme
                    random.setstate(loaded_state['random_state']); np.random.set_state(loaded_state['numpy_random_state'])
                    torch.set_rng_state(loaded_state['torch_random_state'].cpu())
                    logging.info(f"Random states restored from checkpoint (Gen {start_generation}).")
                except Exception as e: logging.warning(f"Could not fully restore random states: {e}")
                initial_state_loaded = True
                logging.info(f"Resuming from Generation {start_generation + 1} with {len(population)} genomes.")
                if tracker_state: logging.info("Innovation tracker state loaded.")
                else: logging.warning("Innovation tracker state not found in checkpoint.")
                if resumed_wandb_id: logging.info(f"Found previous W&B run ID: {resumed_wandb_id}")
            else: logging.error("Failed to load v6 checkpoint. Starting from scratch."); resume_run = False
        elif resume_run: logging.warning(f"Resume requested but no valid v6 checkpoint found. Starting from scratch."); resume_run = False

        # Innovation Tracker Başlatma/Yükleme
        innovation_tracker = InnovationTracker()
        if initial_state_loaded and tracker_state:
            innovation_tracker.set_state(tracker_state)
            logging.info(f"Innovation tracker state restored: NodeID={tracker_state.get('node_id')}, Innov={tracker_state.get('innovation')}")
        else:
             # Sıfırdan başlarken, giriş/çıkış düğümlerini tracker'a kaydetmek iyi olabilir
             # Ancak create_initial_genome zaten ID alıyor. Şimdilik boş başlatalım.
             logging.info("Initialized new innovation tracker.")


        # W&B Başlatma (v5'teki gibi)
        if args.use_wandb and _WANDB_AVAILABLE:
            # ... (v5'teki W&B init kodu buraya gelir, config=args olmalı) ...
            pass # Önceki kodu buraya yapıştırın


        # Config Kaydetme/Loglama (v5'teki gibi)
        # ... (v5'teki config kaydetme/loglama kodu buraya gelir) ...
        pass # Önceki kodu buraya yapıştırın


        # Random Tohum Ayarlama (sadece sıfırdan başlarken)
        if not initial_state_loaded:
             # ... (v5'teki tohum ayarlama kodu buraya gelir) ...
             pass # Önceki kodu buraya yapıştırın


        # Veri Üretimi (v5'teki gibi)
        try:
            logging.info("Generating/Reloading data (v6)...")
            X_train_np, y_train_np = generate_data(args.train_samples, args.seq_length)
            X_test_np, y_test_np = generate_data(args.test_samples, args.seq_length)
            input_size = X_train_np.shape[1]
            output_size = y_train_np.shape[1]
        except Exception: logging.critical("Failed to generate/reload data. Exiting."); sys.exit(1)


        # Popülasyon Başlatma (sadece sıfırdan başlarken - minimal genomlar)
        if not initial_state_loaded:
            logging.info(f"--- Initializing Population (Size: {args.pop_size}) with Minimal Genomes ---")
            try:
                population = [create_initial_genome(input_size, output_size, innovation_tracker) for _ in range(args.pop_size)]
                logging.info("Population initialized successfully.")
            except Exception: logging.critical("Failed to initialize population. Exiting."); sys.exit(1)


        # Evrim Süreci (v6)
        logging.info(f"--- Starting/Resuming PyTorch v6 Evolution ({args.generations} Total Generations) ---")
        best_genome_evolved: Optional[Genome] = None
        best_fitness_hist = [] # TODO: Checkpoint'ten yükle
        avg_fitness_hist = []  # TODO: Checkpoint'ten yükle

        if start_generation >= args.generations:
             logging.warning(f"Loaded checkpoint gen ({start_generation}) >= total gens ({args.generations}). Skipping evolution.")
             # En iyi genomu checkpoint'ten al (TODO: daha iyi yöntem)
             if population: best_genome_evolved = population[0].copy() # En iyiyi seçmek lazım
             else: best_genome_evolved = None
        else:
             try:
                 best_genome_evolved, gen_best_hist, gen_avg_hist = evolve_population_pytorch_v6(
                     population, X_train_np, y_train_np, start_generation, args.generations,
                     args, output_dir, device, innovation_tracker, wandb_run
                 )
                 best_fitness_hist.extend(gen_best_hist)
                 avg_fitness_hist.extend(gen_avg_hist)
             except Exception as e:
                 logging.critical(f"Fatal error during PyTorch v6 evolution process: {e}", exc_info=True)
                 raise

        logging.info("--- PyTorch v6 Evolution Complete ---")

        # Fitness Geçmişi Kaydet/Çizdir (v5'teki gibi)
        # ... (v5'teki plot/save history kodu buraya gelir, dosya adı v6 olmalı) ...
        pass # Önceki kodu buraya yapıştırın

        # En İyi Genomu Değerlendir/Kaydet (Eğitim yok)
        final_metrics = {"test_mse": np.inf, "avg_kendall_tau": 0.0}
        best_genome_details = {}
        saved_best_genome_path = None

        if best_genome_evolved is None:
            logging.error("Evolution did not yield a best genome. Skipping final evaluation.")
        else:
            best_genome_details = {
                 'id': best_genome_evolved.id,
                 'nodes': len(best_genome_evolved.node_genes),
                 'connections': len(best_genome_evolved.connection_genes),
                 'enabled_connections': len([c for c in best_genome_evolved.connection_genes.values() if c.enabled]),
                 'fitness': best_genome_evolved.fitness
            }
            logging.info(f"Best evolved genome details: {best_genome_details}")
            if wandb_run: wandb_run.summary.update({"best_genome_" + k: v for k,v in best_genome_details.items()})

            # Değerlendirme
            final_metrics = evaluate_model_pytorch_v6(best_genome_evolved, X_test_np, y_test_np, args.batch_size, device)
            if wandb_run: wandb_run.summary.update(final_metrics)

            # En iyi genomu kaydet
            saved_best_genome_path = os.path.join(output_dir, "best_genome_v6.json")
            try:
                 with open(saved_best_genome_path, 'w') as f:
                      json.dump(best_genome_evolved.to_dict(), f, indent=4)
                 logging.info(f"Best genome saved to {saved_best_genome_path}")
                 # W&B artifact
                 if wandb_run:
                      try:
                           artifact = wandb.Artifact(f'best_genome_{run_name}', type='genome')
                           artifact.add_file(saved_best_genome_path)
                           wandb_run.log_artifact(artifact)
                           logging.info(f"Saved best genome as W&B artifact.")
                      except Exception as e: logging.warning(f"Failed to save genome as W&B artifact: {e}")
            except Exception as e:
                 logging.error(f"Failed to save best genome: {e}", exc_info=True)
                 saved_best_genome_path = None


        # Sonuçları Kaydet (v6 formatında)
        logging.info("--- Saving Final Results (v6) ---")
        final_results = {
            "run_info": { # ... (v5'teki gibi, version='v6_base') ...
                         },
            "config": vars(args), # Argümanları kaydetmek önemli
            "evolution_summary": { # ... (v5'teki gibi) ...
                                 "best_genome_details": best_genome_details},
            # "final_training_summary": {}, # Eğitim yok
            "final_evaluation_on_test": final_metrics,
            "saved_best_genome_path": saved_best_genome_path
        }
        results_path = os.path.join(output_dir, "final_results_pytorch_v6.json")
        try:
            # ... (v5'teki JSON kaydetme kodu buraya gelir, convert_types güncellenmeli) ...
            pass # Önceki kodu buraya yapıştırın
            logging.info(f"Final v6 results summary saved to {results_path}")
        except Exception as e: logging.error(f"Failed to save final v6 results JSON: {e}", exc_info=True)

    except (Exception, KeyboardInterrupt) as e:
         # ... (v5'teki ana hata yakalama ve W&B bitirme kodu) ...
         pass # Önceki kodu buraya yapıştırın
    finally:
        # ... (v5'teki W&B normal bitirme kodu) ...
        pass # Önceki kodu buraya yapıştırın
        logging.info(f"========== PyTorch v6 Pipeline Run {run_name} Finished ==========")


# --- Argüman Ayrıştırıcı (v6) ---
def parse_arguments_v6() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoNet v6 (Base): Genotype-based Neuroevolution with PyTorch")

    # --- Dizinler, Kontrol, Veri, Paralellik, Eğitim (v5'ten benzer) ---
    # ... (output_base_dir, resume_from, checkpoint_interval, device, seed) ...
    # ... (seq_length, train_samples, test_samples) ...
    # ... (num_workers) ...
    # ... (batch_size, epochs_final_train(kullanılmıyor ama kalabilir), learning_rate(kullanılmıyor)) ...

    # --- Evrim Parametreleri (v6 güncellemeleri) ---
    evo_group = parser.add_argument_group('Evolution Parameters (v6)')
    evo_group.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE)
    evo_group.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS)
    # evo_group.add_argument('--crossover_rate', type=float, default=0.0, help='Crossover not implemented in v6_base.') # Şimdilik 0
    evo_group.add_argument('--tournament_size', type=int, default=DEFAULT_TOURNAMENT_SIZE)
    evo_group.add_argument('--elitism_count', type=int, default=DEFAULT_ELITISM_COUNT)
    evo_group.add_argument('--complexity_penalty', type=float, default=DEFAULT_COMPLEXITY_PENALTY)

    # --- Mutasyon Oranları (v6) ---
    mut_group = parser.add_argument_group('Mutation Rates (v6)')
    mut_group.add_argument('--conn_mut_rate', type=float, default=DEFAULT_CONN_MUT_RATE, help="Prob. to mutate existing connection weights.")
    mut_group.add_argument('--bias_mut_rate', type=float, default=DEFAULT_BIAS_MUT_RATE, help="Prob. to mutate existing node biases.")
    mut_group.add_argument('--add_conn_rate', type=float, default=DEFAULT_ADD_CONN_RATE, help="Prob. to add a new connection.")
    mut_group.add_argument('--add_node_rate', type=float, default=DEFAULT_ADD_NODE_RATE, help="Prob. to add a new node by splitting a connection.")
    mut_group.add_argument('--toggle_enable_rate', type=float, default=DEFAULT_TOGGLE_ENABLE_RATE, help="Prob. to toggle a connection's enabled state.")
    mut_group.add_argument('--activation_mut_rate', type=float, default=DEFAULT_ACTIVATION_MUT_RATE, help="Prob. to mutate a hidden node's activation function.")

    # --- Adaptif Mutasyon (Ağırlık/Bias için hala geçerli) ---
    adapt_group = parser.add_argument_group('Adaptive Mutation (for Weights/Biases)')
    # ... (v5'teki adapt_mutation, mutation_strength, stagnation_limit vb. argümanlar) ...

    # --- Deney Takibi (W&B) ---
    wandb_group = parser.add_argument_group('Experiment Tracking (Weights & Biases)')
    # ... (v5'teki use_wandb, wandb_project, wandb_entity argümanları) ...

    # Önceki argümanları kopyalamak yerine, v5 parser'ını temel alıp güncellemek daha iyi.
    # Yukarıdaki yorumlar yerine gerçek argüman tanımları eklenmeli.
    # ŞİMDİLİK YER TUTUCU - GERÇEK ARG PARSER KODU GEREKİR
    print("WARNING: Argument parser needs to be fully implemented based on v5 and v6 additions.")
    args = parser.parse_args() # Bu satır hata verir, parser tanımları eksik.
    if args.seed is None: args.seed = random.randint(0, 2**32 - 1); print(f"Generated random seed: {args.seed}")
    return args


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # Gerçek Argümanları al
    # cli_args = parse_arguments_v6() # Tamamlanmış parser fonksiyonu çağrılmalı
    # run_pipeline_pytorch_v6(cli_args)
    print("\nERROR: EvoNet v6 (Step 1) code structure provided.")
    print("ERROR: Argument parser and integration of helper functions (logging, device, data, plotting, saving) need completion based on v5 code.")
    print("ERROR: Please integrate the provided v6 classes and logic into the full v5 pipeline structure.")
    # Örnek kullanım için (parser olmadan):
    # fake_args = argparse.Namespace( ... varsayılan değerler ... )
    # run_pipeline_pytorch_v6(fake_args)