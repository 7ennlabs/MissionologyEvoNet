# ==============================================================================
# EvoNet Optimizer 2 - Revize Edilmiş ve İyileştirilmiş Kod
# Açıklama: Bu kod, sıralama görevini öğrenmek için rastgele topolojilere
# sahip sinir ağlarını evrimleştiren bir neuroevolution süreci uygular.
# Daha sağlam hata kontrolü, yapılandırma, loglama ve iyileştirilmiş
# evrimsel operatörler içerir.
# ==============================================================================

import os
import subprocess
import sys
import argparse
import random
import logging
from datetime import datetime
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# --- Sabitler ve Varsayılan Değerler ---
DEFAULT_SEQ_LENGTH = 10
DEFAULT_POP_SIZE = 50
DEFAULT_GENERATIONS = 50
DEFAULT_MUTATION_RATE = 0.4       # Bireye mutasyon uygulama olasılığı
DEFAULT_WEIGHT_MUT_RATE = 0.8     # Mutasyon olursa, ağırlık bozulması olasılığı
DEFAULT_ACTIVATION_MUT_RATE = 0.2 # Mutasyon olursa, aktivasyon değişimi olasılığı
DEFAULT_MUTATION_STRENGTH = 0.1 # Ağırlık bozulmasının büyüklüğü (std dev)
DEFAULT_TOURNAMENT_SIZE = 5       # Turnuva seçilimindeki birey sayısı
DEFAULT_ELITISM_COUNT = 2         # Sonraki nesle doğrudan aktarılacak en iyi birey sayısı
DEFAULT_EPOCHS_FINAL_TRAIN = 100  # En iyi modelin son eğitimindeki max epoch
DEFAULT_BATCH_SIZE = 64           # Tahmin ve eğitim için batch boyutu
DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "evonet_runs_revised") # Ana çıktı klasörü

# --- Loglama Ayarları ---
def setup_logging(log_dir: str, log_level=logging.INFO) -> None:
    """Loglamayı dosyaya ve konsola ayarlayan fonksiyon."""
    log_filename = os.path.join(log_dir, 'evolution_run.log')
    # Önceki handler'ları temizle (Jupyter gibi ortamlarda tekrar çalıştırmada önemli)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Yeni handler'ları ayarla
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'), # 'w' modu ile her çalıştırmada üzerine yazar
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging setup complete.")

# --- GPU Kontrolü ---
def check_gpu() -> bool:
    """GPU varlığını kontrol eder ve bellek artışını ayarlar."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found.")
            if logical_gpus:
                 logging.info(f"Using GPU: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")
            return True
        except RuntimeError as e:
            logging.error(f"Error setting memory growth for GPU: {e}", exc_info=True)
            return False
    else:
        logging.warning("GPU not found. Using CPU.")
        return False

# --- Veri Üretimi ---
def generate_data(num_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rastgele diziler ve sıralanmış hallerini üretir."""
    logging.info(f"Generating {num_samples} samples with sequence length {seq_length}...")
    try:
        X = np.random.rand(num_samples, seq_length).astype(np.float32) * 100
        y = np.sort(X, axis=1).astype(np.float32)
        logging.info("Data generation successful.")
        return X, y
    except Exception as e:
        logging.error(f"Error during data generation: {e}", exc_info=True)
        raise # Hatanın yukarıya bildirilmesi önemli

# --- Neuroevolution Çekirdeği ---
def create_individual(seq_length: int, input_shape: Tuple) -> Sequential:
    """Rastgele mimariye sahip bir Keras Sequential modeli oluşturur ve derler."""
    try:
        model = Sequential(name=f"model_random_{random.randint(10000, 99999)}")
        num_hidden_layers = random.randint(1, 4)
        neurons_per_layer = [random.randint(8, 64) for _ in range(num_hidden_layers)]
        activations = [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(num_hidden_layers)]

        model.add(Input(shape=input_shape)) # Input katmanı

        for i in range(num_hidden_layers): # Gizli katmanlar
            model.add(Dense(neurons_per_layer[i], activation=activations[i]))

        model.add(Dense(seq_length, activation='linear')) # Çıkış katmanı

        # Ağırlık manipülasyonu ve potansiyel eğitim için modeli derle
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        #logging.debug(f"Created individual: {model.name} with {len(model.layers)} layers.")
        return model
    except Exception as e:
        logging.error(f"Error creating individual model: {e}", exc_info=True)
        raise

@tf.function # TensorFlow grafiği olarak derleyerek potansiyel hızlandırma
def get_predictions(model: Sequential, X: tf.Tensor) -> tf.Tensor:
    """Model tahminlerini tf.function kullanarak alır."""
    return model(X, training=False)

def calculate_fitness(individual: Sequential, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    """Bir bireyin fitness değerini (1/MSE) hesaplar, hataları yönetir."""
    if not isinstance(X, tf.Tensor): X = tf.cast(X, tf.float32)
    if not isinstance(y, tf.Tensor): y = tf.cast(y, tf.float32)

    try:
        y_pred_tf = get_predictions(individual, X) # Batching predict içinde yapılır
        mse = tf.reduce_mean(tf.square(y - y_pred_tf))
        mse_val = mse.numpy()

        # Fitness: Ters MSE (sıfıra bölmeyi önlemek için epsilon ekle)
        fitness_score = 1.0 / (mse_val + 1e-8)

        if not np.isfinite(fitness_score) or fitness_score < 0:
            logging.warning(f"Non-finite or negative fitness detected ({fitness_score:.4g}) for model {individual.name}. Assigning minimal fitness.")
            return 1e-8 # Çok düşük bir fitness ata

        #logging.debug(f"Fitness for {individual.name}: {fitness_score:.4f} (MSE: {mse_val:.4f})")
        return float(fitness_score)

    except tf.errors.InvalidArgumentError as e:
         logging.error(f"TensorFlow InvalidArgumentError during fitness calculation for model {individual.name} (potential shape mismatch?): {e}")
         return 1e-8
    except Exception as e:
        logging.error(f"Unhandled error during fitness calculation for model {individual.name}: {e}", exc_info=True)
        return 1e-8 # Hata durumunda minimum fitness döndür


def mutate_individual(individual: Sequential, weight_mut_rate: float, act_mut_rate: float, mut_strength: float) -> Sequential:
    """Bir bireye mutasyonlar uygular (ağırlık bozulması, aktivasyon değişimi)."""
    try:
        # Mutasyon için modeli klonla, orijinali bozma
        mutated_model = clone_model(individual)
        mutated_model.set_weights(individual.get_weights())

        mutated = False
        # 1. Ağırlık Mutasyonu
        if random.random() < weight_mut_rate:
            mutated = True
            for layer in mutated_model.layers:
                if isinstance(layer, Dense) and layer.get_weights(): # Sadece ağırlığı olan Dense katmanları
                    weights_biases = layer.get_weights()
                    new_weights_biases = []
                    for wb in weights_biases:
                        noise = np.random.normal(0, mut_strength, wb.shape).astype(np.float32)
                        new_weights_biases.append(wb + noise)
                    layer.set_weights(new_weights_biases)

        # 2. Aktivasyon Mutasyonu (Bağımsız olasılık)
        if random.random() < act_mut_rate:
            dense_layers = [layer for layer in mutated_model.layers if isinstance(layer, Dense)]
            if len(dense_layers) > 1: # En az bir gizli katman varsa
                layer_to_mutate = random.choice(dense_layers[:-1]) # Çıkış katmanı hariç
                current_activation_name = tf.keras.activations.serialize(layer_to_mutate.activation)
                possible_activations = ['relu', 'tanh', 'sigmoid']
                if current_activation_name in possible_activations:
                     possible_activations.remove(current_activation_name)
                if possible_activations: # Değiştirilecek başka aktivasyon varsa
                    new_activation = random.choice(possible_activations)
                    # Katman konfigürasyonunu güncellemek daha güvenli
                    layer_config = layer_to_mutate.get_config()
                    layer_config['activation'] = new_activation
                    # Yeni konfigürasyondan yeni katman oluştur ve ağırlıkları aktar
                    try:
                        new_layer = Dense.from_config(layer_config)
                        # Model içinde katmanı değiştirmek yerine, modeli yeniden oluşturmak daha sağlam olabilir.
                        # Ancak basitlik için bu yaklaşımı deneyelim (riskli olabilir).
                        # Aktivasyon değiştirmek için katmanı yeniden build etmek gerekebilir.
                        # Bu kısım karmaşık olabilir, şimdilik loglayalım.
                        logging.debug(f"Attempting activation change on layer {layer_to_mutate.name} to {new_activation} (Implementation needs robust handling).")
                        # Gerçek uygulamada modeli yeniden oluşturmak daha iyi olabilir.
                        # Şimdilik sadece ağırlık mutasyonuna odaklanalım. Aktivasyon mutasyonu deneysel kalabilir.
                        mutated = True # Aktivasyon mutasyon denemesi yapıldı olarak işaretle
                    except Exception as e:
                        logging.warning(f"Could not directly modify/rebuild layer for activation change: {e}")


        # Mutasyona uğradıysa modeli yeniden derle (optimizer durumu sıfırlanabilir)
        if mutated:
            mutated_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            mutated_model._name = f"mutated_{individual.name}_{random.randint(1000,9999)}" # İsmi güncelle
            #logging.debug(f"Mutated model {individual.name} -> {mutated_model.name}")

        return mutated_model
    except Exception as e:
        logging.error(f"Error during mutation of model {individual.name}: {e}", exc_info=True)
        return individual # Hata olursa orijinal bireyi döndür


def tournament_selection(population: List[Sequential], fitness_scores: List[float], k: int) -> Sequential:
    """Rastgele seçilen bir turnuva grubundan en iyi bireyi seçer."""
    if not population:
        raise ValueError("Population cannot be empty for selection.")
    if len(population) < k:
        logging.warning(f"Tournament size {k} is larger than population size {len(population)}. Using population size.")
        k = len(population)
    try:
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_local_idx = np.argmax(tournament_fitness)
        winner_global_idx = tournament_indices[winner_local_idx]
        #logging.debug(f"Tournament winner: Index {winner_global_idx}, Fitness: {fitness_scores[winner_global_idx]:.4f}")
        return population[winner_global_idx]
    except Exception as e:
        logging.error(f"Error during tournament selection: {e}", exc_info=True)
        # Hata durumunda rastgele bir birey seçmek bir alternatif olabilir
        return random.choice(population)


def evolve_population(population: List[Sequential], X: np.ndarray, y: np.ndarray, generations: int,
                      mutation_rate: float, weight_mut_rate: float, act_mut_rate: float, mut_strength: float,
                      tournament_size: int, elitism_count: int, batch_size: int) -> Tuple[Sequential, List[float], List[float]]:
    """Evrimsel süreci çalıştırır, en iyi modeli ve fitness geçmişini döndürür."""
    best_fitness_history = []
    avg_fitness_history = []
    best_model_overall = None
    best_fitness_overall = -np.inf # Negatif sonsuz ile başla

    # Veriyi TensorFlow tensor'üne dönüştür (döngü dışında bir kez yap)
    X_tf = tf.cast(X, tf.float32)
    y_tf = tf.cast(y, tf.float32)

    for gen in range(generations):
        generation_start_time = datetime.now()
        # 1. Fitness Değerlendirme
        try:
            # Tüm popülasyon için fitness'ı hesapla
            fitness_scores = [calculate_fitness(ind, X_tf, y_tf, batch_size) for ind in population]
        except Exception as e:
            logging.critical(f"Error calculating fitness for population in Generation {gen+1}: {e}", exc_info=True)
            # Bu kritik bir hata, süreci durdurmak gerekebilir veya önceki popülasyonla devam edilebilir.
            # Şimdilik en iyi modeli döndürelim ve çıkalım.
            if best_model_overall: return best_model_overall, best_fitness_history, avg_fitness_history
            else: raise # Henüz iyi model yoksa hatayı yükselt

        # 2. İstatistikler ve En İyiyi Takip Etme
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        avg_fitness = np.mean(fitness_scores)
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)

        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            try:
                # En iyi modelin yapısını ve ağırlıklarını güvenli bir şekilde kopyala
                best_model_overall = clone_model(population[current_best_idx])
                best_model_overall.set_weights(population[current_best_idx].get_weights())
                best_model_overall.compile(optimizer=Adam(), loss='mse') # Yeniden derle
                logging.info(f"Generation {gen+1}: *** New overall best fitness found: {best_fitness_overall:.6f} ***")
            except Exception as e:
                 logging.error(f"Could not clone or set weights for the new best model: {e}", exc_info=True)
                 # Klonlama başarısız olursa devam et, ama en iyi model güncellenmemiş olabilir.
                 best_fitness_overall = current_best_fitness # Fitness'ı yine de güncelle

        generation_time = (datetime.now() - generation_start_time).total_seconds()
        logging.info(f"Generation {gen+1}/{generations} | Best Fitness: {current_best_fitness:.6f} | Avg Fitness: {avg_fitness:.6f} | Time: {generation_time:.2f}s")

        # 3. Yeni Popülasyon Oluşturma
        new_population = []

        # 3a. Elitizm
        if elitism_count > 0 and len(population) >= elitism_count:
            try:
                elite_indices = np.argsort(fitness_scores)[-elitism_count:]
                for idx in elite_indices:
                    elite_clone = clone_model(population[idx])
                    elite_clone.set_weights(population[idx].get_weights())
                    elite_clone.compile(optimizer=Adam(), loss='mse')
                    new_population.append(elite_clone)
                    #logging.debug(f"Added elite model {elite_clone.name} (Index: {idx}, Fitness: {fitness_scores[idx]:.4f})")
            except Exception as e:
                 logging.error(f"Error during elitism: {e}", exc_info=True)


        # 3b. Seçilim ve Üreme (Kalan Bireyler İçin)
        num_to_generate = len(population) - len(new_population)
        offspring_population = []
        while len(offspring_population) < num_to_generate:
            try:
                # Ebeveyn seç
                parent = tournament_selection(population, fitness_scores, tournament_size)

                # Çocuk oluştur (mutasyon uygula veya uygulama)
                if random.random() < mutation_rate:
                    child = mutate_individual(parent, weight_mut_rate, act_mut_rate, mut_strength)
                else:
                    # Mutasyon yoksa, yine de klonla ki aynı nesne referansı olmasın
                    child = clone_model(parent)
                    child.set_weights(parent.get_weights())
                    child.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                    child._name = f"cloned_{parent.name}_{random.randint(1000,9999)}" # İsmi güncelle

                offspring_population.append(child)
            except Exception as e:
                logging.error(f"Error during selection/reproduction cycle: {e}", exc_info=True)
                # Hata durumunda döngüyü kırmak veya rastgele birey eklemek düşünülebilir
                # Şimdilik döngü devam etsin, belki sonraki denemede düzelir
                if len(offspring_population) < num_to_generate: # Eksik kalmaması için rastgele ekle
                    logging.warning("Adding random individual due to reproduction error.")
                    offspring_population.append(create_individual(y.shape[1], X.shape[1:]))


        new_population.extend(offspring_population)
        population = new_population # Popülasyonu güncelle

    # Döngü bittiğinde en iyi modeli döndür
    if best_model_overall is None and population: # Hiç iyileşme olmadıysa veya elitizm yoksa
         logging.warning("No overall best model tracked (or cloning failed). Returning best from final population.")
         final_fitness_scores = [calculate_fitness(ind, X_tf, y_tf, batch_size) for ind in population]
         best_idx_final = np.argmax(final_fitness_scores)
         best_model_overall = population[best_idx_final]
    elif not population:
         logging.error("Evolution finished with an empty population!")
         return None, best_fitness_history, avg_fitness_history


    logging.info(f"Evolution finished. Best fitness achieved: {best_fitness_overall:.6f}")
    return best_model_overall, best_fitness_history, avg_fitness_history


# --- Grafik Çizimi ---
def plot_fitness_history(history_best: List[float], history_avg: List[float], output_dir: str) -> None:
    """Fitness geçmişini çizer ve kaydeder."""
    if not history_best or not history_avg:
        logging.warning("Fitness history is empty, cannot plot.")
        return
    try:
        plt.figure(figsize=(12, 7))
        plt.plot(history_best, label="Best Fitness per Generation", marker='o', linestyle='-', linewidth=2)
        plt.plot(history_avg, label="Average Fitness per Generation", marker='x', linestyle='--', alpha=0.7)
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score (1 / MSE)")
        plt.title("Evolutionary Process Fitness History")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "fitness_history.png")
        plt.savefig(plot_path)
        plt.close() # Bellekte figürü kapat
        logging.info(f"Fitness history plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Error plotting fitness history: {e}", exc_info=True)

# --- Değerlendirme ---
def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, batch_size: int) -> Dict[str, float]:
    """Son modeli test verisi üzerinde değerlendirir."""
    if model is None:
        logging.error("Cannot evaluate a None model.")
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0}
    logging.info("Evaluating final model on test data...")
    try:
        y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
        test_mse = np.mean(np.square(y_test - y_pred))
        logging.info(f"Final Test MSE: {test_mse:.6f}")

        # Kendall's Tau (örneklem üzerinde)
        sample_size = min(500, X_test.shape[0]) # Örneklem boyutunu ayarla
        taus = []
        indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
        for i in indices:
            try:
                tau, _ = kendalltau(y_test[i], y_pred[i])
                if not np.isnan(tau): taus.append(tau)
            except ValueError as ve: # Eğer y_pred sabit değerler içeriyorsa
                 logging.debug(f"Kendall tau ValueError for sample {i}: {ve}")

        avg_kendall_tau = np.mean(taus) if taus else 0.0
        logging.info(f"Average Kendall's Tau (on {sample_size} samples): {avg_kendall_tau:.4f}")

        return {
            "test_mse": float(test_mse),
            "avg_kendall_tau": float(avg_kendall_tau)
        }
    except Exception as e:
        logging.error(f"Error during final model evaluation: {e}", exc_info=True)
        return {"test_mse": np.inf, "avg_kendall_tau": 0.0} # Hata durumunda kötü değerler döndür

# --- Ana İş Akışı ---
def run_pipeline(args: argparse.Namespace):
    """Tüm neuroevolution iş akışını çalıştırır."""

    # Benzersiz çıktı klasörü oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evorun_{timestamp}_gen{args.generations}_pop{args.pop_size}"
    output_dir = os.path.join(args.output_base_dir, run_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"FATAL: Could not create output directory: {output_dir}. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Loglamayı ayarla
    setup_logging(output_dir)
    logging.info(f"========== Starting EvoNet Pipeline Run: {run_name} ==========")
    logging.info(f"Output directory: {output_dir}")

    # Argümanları logla ve kaydet
    logging.info("--- Configuration ---")
    args_dict = vars(args)
    for k, v in args_dict.items():
        logging.info(f"  {k:<20}: {v}")
    logging.info("---------------------")
    config_path = os.path.join(output_dir, "config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(args_dict, f, indent=4, sort_keys=True)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration: {e}", exc_info=True)


    # Rastgele tohumları ayarla
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        logging.info(f"Using random seed: {args.seed}")
        # Deterministic ops (TensorFlow >= 2.8): Opsiyonel, performansı düşürebilir ama tekrarlanabilirliği artırır
        # tf.config.experimental.enable_op_determinism()
    except Exception as e:
         logging.warning(f"Could not set all random seeds: {e}")


    # GPU kontrolü
    is_gpu_available = check_gpu()

    # Veri Üretimi
    try:
        X_train, y_train = generate_data(args.train_samples, args.seq_length)
        X_test, y_test = generate_data(args.test_samples, args.seq_length)
        input_shape = X_train.shape[1:] # Model oluşturmak için girdi şekli
    except Exception:
        logging.critical("Failed to generate data. Exiting.")
        sys.exit(1)


    # Popülasyon Başlatma
    logging.info(f"--- Initializing Population (Size: {args.pop_size}) ---")
    try:
        population = [create_individual(args.seq_length, input_shape) for _ in range(args.pop_size)]
        logging.info("Population initialized successfully.")
    except Exception:
        logging.critical("Failed to initialize population. Exiting.")
        sys.exit(1)

    # Evrim Süreci
    logging.info(f"--- Starting Evolution ({args.generations} Generations) ---")
    try:
        best_model_unevolved, best_fitness_hist, avg_fitness_hist = evolve_population(
            population, X_train, y_train, args.generations,
            args.mutation_rate, args.weight_mut_rate, args.activation_mut_rate, args.mutation_strength,
            args.tournament_size, args.elitism_count, args.batch_size
        )
    except Exception as e:
        logging.critical(f"Fatal error during evolution process: {e}", exc_info=True)
        sys.exit(1)
    logging.info("--- Evolution Complete ---")

    # Fitness geçmişini kaydet ve çizdir
    if best_fitness_hist and avg_fitness_hist:
        history_path = os.path.join(output_dir, "fitness_history.csv")
        try:
            history_data = np.array([np.arange(1, len(best_fitness_hist) + 1), best_fitness_hist, avg_fitness_hist]).T
            np.savetxt(history_path, history_data, delimiter=',', header='Generation,BestFitness,AvgFitness', comments='', fmt=['%d', '%.8f', '%.8f'])
            logging.info(f"Fitness history data saved to {history_path}")
        except Exception as e:
            logging.error(f"Could not save fitness history data: {e}", exc_info=True)
        plot_fitness_history(best_fitness_hist, avg_fitness_hist, output_dir)
    else:
        logging.warning("Fitness history is empty, skipping saving/plotting.")


    # En İyi Modelin Son Eğitimi
    if best_model_unevolved is None:
        logging.error("Evolution did not yield a best model. Skipping final training and evaluation.")
        final_metrics = {"test_mse": np.inf, "avg_kendall_tau": 0.0}
        final_model_path = None
        training_summary = {}
    else:
        logging.info("--- Starting Final Training of Best Evolved Model ---")
        try:
            # En iyi modeli tekrar klonla ve derle (güvenlik için)
            final_model = clone_model(best_model_unevolved)
            final_model.set_weights(best_model_unevolved.get_weights())
            # Son eğitim için belki farklı bir öğrenme oranı denenebilir
            final_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            logging.info("Model Summary of Best Evolved (Untrained):")
            final_model.summary(print_fn=logging.info)


            # Callback'ler
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1) # Sabrı biraz artır
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, min_lr=1e-7, verbose=1) # Faktörü ve sabrı ayarla

            history = final_model.fit(
                X_train, y_train,
                epochs=args.epochs_final_train,
                batch_size=args.batch_size,
                validation_split=0.2, # Eğitim verisinin %20'si validasyon için
                callbacks=[early_stopping, reduce_lr],
                verbose=2 # Her epoch için bir satır log
            )
            logging.info("Final training complete.")
            training_summary = {
                 "epochs_run": len(history.history['loss']),
                 "final_train_loss": history.history['loss'][-1],
                 "final_val_loss": history.history['val_loss'][-1]
            }

            # Eğitilmiş modeli değerlendir
            final_metrics = evaluate_model(final_model, X_test, y_test, args.batch_size)

            # Eğitilmiş modeli kaydet
            final_model_path = os.path.join(output_dir, "best_evolved_model_trained.keras")
            final_model.save(final_model_path)
            logging.info(f"Final trained model saved to {final_model_path}")

        except Exception as e:
             logging.error(f"Error during final training or evaluation: {e}", exc_info=True)
             final_metrics = {"test_mse": np.inf, "avg_kendall_tau": 0.0}
             final_model_path = None
             training_summary = {"error": str(e)}


    # Sonuçları Kaydet
    logging.info("--- Saving Final Results ---")
    final_results = {
        "run_info": {
            "run_name": run_name,
            "timestamp": timestamp,
            "output_directory": output_dir,
            "gpu_used": is_gpu_available,
        },
        "config": args_dict,
        "evolution_summary": {
            "generations_run": len(best_fitness_hist) if best_fitness_hist else 0,
            "best_fitness_achieved": best_fitness_overall if best_fitness_overall > -np.inf else None,
            "best_fitness_final_gen": best_fitness_hist[-1] if best_fitness_hist else None,
            "avg_fitness_final_gen": avg_fitness_hist[-1] if avg_fitness_hist else None,
        },
         "final_training_summary": training_summary,
        "final_evaluation_on_test": final_metrics,
        "saved_model_path": final_model_path
    }
    results_path = os.path.join(output_dir, "final_results.json")
    try:
        # JSON'a kaydederken NumPy türlerini dönüştür
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4, default=convert_numpy_types) # default handler ekle
        logging.info(f"Final results summary saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save final results JSON: {e}", exc_info=True)

    logging.info(f"========== Pipeline Run {run_name} Finished ==========")


# --- Argüman Ayrıştırıcı ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoNet Revised: Neuroevolution for Sorting Task")

    # --- Dizinler ---
    parser.add_argument('--output_base_dir', type=str, default=DEFAULT_OUTPUT_BASE_DIR,
                        help='Base directory to store run results.')

    # --- Veri Ayarları ---
    parser.add_argument('--seq_length', type=int, default=DEFAULT_SEQ_LENGTH, help='Length of sequences.')
    parser.add_argument('--train_samples', type=int, default=5000, help='Number of training samples.')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples.')

    # --- Evrim Parametreleri ---
    parser.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE, help='Population size.')
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS, help='Number of generations.')
    parser.add_argument('--mutation_rate', type=float, default=DEFAULT_MUTATION_RATE, help='Overall mutation probability.')
    parser.add_argument('--weight_mut_rate', type=float, default=DEFAULT_WEIGHT_MUT_RATE, help='Weight mutation probability (if mutation occurs).')
    parser.add_argument('--activation_mut_rate', type=float, default=DEFAULT_ACTIVATION_MUT_RATE, help='Activation mutation probability (if mutation occurs).')
    parser.add_argument('--mutation_strength', type=float, default=DEFAULT_MUTATION_STRENGTH, help='Std dev for weight mutation noise.')
    parser.add_argument('--tournament_size', type=int, default=DEFAULT_TOURNAMENT_SIZE, help='Number of individuals in tournament selection.')
    parser.add_argument('--elitism_count', type=int, default=DEFAULT_ELITISM_COUNT, help='Number of elite individuals to carry over.')

    # --- Eğitim ve Değerlendirme ---
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for predictions and final training.')
    parser.add_argument('--epochs_final_train', type=int, default=DEFAULT_EPOCHS_FINAL_TRAIN, help='Max epochs for final training.')

    # --- Tekrarlanabilirlik ---
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: random).')

    args = parser.parse_args()

    # Varsayılan tohum ayarla (eğer verilmediyse)
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {args.seed}") # Loglama başlamadan önce print et

    return args


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # Argümanları ayrıştır
    cli_args = parse_arguments()

    # Ana iş akışını çalıştır
    try:
        run_pipeline(cli_args)
    except SystemExit: # sys.exit() çağrılarını yakala ve normal çıkış yap
        pass
    except Exception as e:
        # Loglama başlamamışsa bile hatayı yazdırmaya çalış
        print(f"\nFATAL UNHANDLED ERROR in main execution block: {e}", file=sys.stderr)
        # Loglama ayarlandıysa oraya da yaz
        if logging.getLogger().hasHandlers():
             logging.critical("FATAL UNHANDLED ERROR in main execution block:", exc_info=True)
        else:
             import traceback
             print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1) # Hata kodu ile çık