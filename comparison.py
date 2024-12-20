import gym  # OpenAI Gym kütüphanesi, simülasyon ortamları oluşturmak için kullanılır
import numpy as np  # Sayısal işlemler ve diziler için kütüphane
import torch  # Tensor hesaplamaları ve derin öğrenme modelleri için
import torch.nn as nn  # Sinir ağı modüllerini tanımlamak için kullanılır
import torch.optim as optim  # Optimizasyon algoritmalarını kullanmak için
import matplotlib.pyplot as plt  # Grafik sonuçlarını çizmek için

# Hesaplamalar için cihaz ayarı (GPU varsa kullanılır, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ajanlar için sinir ağı mimarisini tanımlıyoruz
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(NeuralNetwork, self).__init__()  # Üst sınıfın (nn.Module) __init__ metodunu çağırıyoruz
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # İlk tam bağlantılı katman
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # İkinci tam bağlantılı katman
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Çıkış katmanı

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # İlk katmanda ReLU aktivasyonu uygulanıyor
        x = torch.relu(self.fc2(x))  # İkinci katmanda ReLU aktivasyonu uygulanıyor
        return self.fc3(x)  # Sonuç döndürülüyor

# PPO (Proximal Policy Optimization) ajanının tanımı
class PPOAgent:
    def __init__(self, env):
        self.env = env  # Ortamı saklıyoruz
        # Gözlem ve eylem boyutlarını ortamdan alıyoruz
        obs_dim = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.Box) else 1
        act_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n

        # Aktör (Politika ağı) ve Eleştirmen (Değer ağı) modelleri oluşturuluyor
        self.actor = NeuralNetwork(obs_dim, act_dim).to(device)
        self.critic = NeuralNetwork(obs_dim, 1).to(device)
        # Aktör ve Eleştirmen ağları için optimizasyon algoritmaları
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = 0.99  # İndirgeme faktörü
        self.eps_clip = 0.2  # PPO algoritması için klip oranı

    def select_action(self, state):
        # Durumu tensöre çevir ve cihazda işleme hazır hale getir
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean = self.actor(state)  # Aktörden eylem ortalamasını al
        if isinstance(self.env.action_space, gym.spaces.Box):
            # Sürekli aksiyon alanı için eylemleri -1 ile 1 arasına sıkıştır
            action = torch.tanh(action_mean).detach().cpu().numpy()[0]
        else:
            # Ayrık aksiyonlar için olasılıkları hesapla ve bir eylem seç
            action_prob = torch.softmax(action_mean, dim=-1)
            action = torch.multinomial(action_prob, 1).item()
        return action

    def train(self, trajectories):
        # Toplananlar üzerinde eğitim yap
        for states, actions, rewards in trajectories:
            states = torch.tensor(states, dtype=torch.float32).to(device)  # Gözlemleri tensöre çevir
            actions = torch.tensor(actions, dtype=torch.float32).to(device)  # Eylemleri tensöre çevir
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # Ödülleri tensöre çevir

            # Eleştirmen modelinden durum değerlerini al
            values = self.critic(states).squeeze()
            advantages = rewards - values.detach()  # Avantaj hesapla
            critic_loss = advantages.pow(2).mean()  # Eleştirmen için kayıp fonksiyonu 

            # Eleştirmen modelini güncelle
            self.critic_optimizer.zero_grad()  # Gradients sıfırlanır
            critic_loss.backward()  # Loss için geri yayılım yapılır
            self.critic_optimizer.step()  # Parametreler güncellenir

            action_means = self.actor(states)  # Aktörden tahmin edilen eylemleri al
            if isinstance(self.env.action_space, gym.spaces.Box):
                # Sürekli aksiyon alanı için Gaussian log olasılığı
                log_probs = -((actions - action_means).pow(2)).sum(axis=1)
            else:
                # Ayrık aksiyon alanı için kategorik log olasılığı
                log_probs = torch.log_softmax(action_means, dim=-1)[range(len(actions)), actions]

            actor_loss = -(log_probs * advantages.detach()).mean()  # Aktör kaybı hesaplanır

            # Aktör modelini güncelle
            self.actor_optimizer.zero_grad()  # Gradients sıfırlanır
            actor_loss.backward()  # Loss için geri yayılım yapılır
            self.actor_optimizer.step()  # Parametreler güncellenir
# DDPG (Deep Deterministic Policy Gradient) ajanının tanımı
class DDPGAgent:
    def __init__(self, env):
        # Eğer aksiyon alanı sürekli değilse, hata ver
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("DDPG yalnızca sürekli aksiyon alanlarını destekler.")

        self.env = env  # Ortamı sakla
        obs_dim = env.observation_space.shape[0]  # Gözlem boyutunu belirle
        act_dim = env.action_space.shape[0]  # Aksiyon boyutunu belirle

        # Aktör ve Eleştirmen ağları ile bunların hedef ağlarını oluştur
        self.actor = NeuralNetwork(obs_dim, act_dim).to(device)
        self.critic = NeuralNetwork(obs_dim + act_dim, 1).to(device)
        self.target_actor = NeuralNetwork(obs_dim, act_dim).to(device)
        self.target_critic = NeuralNetwork(obs_dim + act_dim, 1).to(device)

        # Optimizasyon algoritmaları
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Hiperparametreler
        self.gamma = 0.99  # İndirgeme faktörü
        self.tau = 0.005  # Hedef ağlar için güncelleme katsayısı

    def select_action(self, state):
        # Durumu tensöre çevir ve cihazda işleme hazır hale getir
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.tanh(self.actor(state)).detach().cpu().numpy()[0]  # Sürekli aksiyon seç
        return action

    def train(self, buffer):
        pass  

# DQL (Deep Q-Learning) ajanının tanımı
class DQLAgent:
    def __init__(self, env):
        # Eğer aksiyon alanı ayrık değilse, hata ver
        if not hasattr(env.action_space, 'n'):
            raise ValueError("DQL yalnızca ayrık aksiyon alanlarını destekler.")

        self.env = env  # Ortamı sakla
        obs_dim = env.observation_space.shape[0]  # Gözlem boyutunu belirle
        act_dim = env.action_space.n  # Ayrık aksiyonların sayısını belirle

        # Q-ağı ve hedef Q-ağını oluştur
        self.q_network = NeuralNetwork(obs_dim, act_dim).to(device)
        self.target_q_network = NeuralNetwork(obs_dim, act_dim).to(device)

        # Optimizasyon algoritması
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        # Hiperparametreler
        self.gamma = 0.99  # İndirgeme faktörü
        self.epsilon = 1.0  # Başlangıç keşif oranı
        self.epsilon_decay = 0.995  # Keşif oranı için azalma hızı
        self.epsilon_min = 0.1  # Minimum keşif oranı

    def select_action(self, state):
        # Epsilon-greedy politikası
        if np.random.rand() < self.epsilon:  # Keşif yapılacak mı?
            return self.env.action_space.sample()  # Rastgele aksiyon seç
        # Durumu tensöre çevir ve cihaza taşı
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.q_network(state)  # Q-değerlerini hesapla
        return torch.argmax(q_values).item()  # En yüksek Q-değerine sahip aksiyonu seç

    def train(self, buffer):
        pass  # Eğitim mantığı burada tanımlanacak (şu an boş)

# Acrobot ortamında PPO eğitimi fonksiyonu
def train_ppo_acrobot():
    return train_agent("Acrobot-v1", "PPO", episodes=500)  # 500 bölüm boyunca eğitim

# Acrobot ortamında DQL eğitimi fonksiyonu
def train_dql_acrobot():
    return train_agent("Acrobot-v1", "DQL", episodes=500)  # 500 bölüm boyunca eğitim

# MountainCarContinuous ortamında DDPG eğitimi fonksiyonu
def train_ddpg_mountaincar():
    return train_agent("MountainCarContinuous-v0", "DDPG", episodes=500)  # 500 bölüm boyunca eğitim

# MountainCarContinuous ortamında PPO eğitimi fonksiyonu
def train_ppo_mountaincar():
    return train_agent("MountainCarContinuous-v0", "PPO", episodes=500)  # 500 bölüm boyunca eğitim

# Sonuçları karşılaştırmak için grafik çizme fonksiyonu
def plot_results(results, env_name):
    plt.figure()  # Yeni bir grafik figürü oluştur
    for algo, rewards in results.items():
        if env_name in algo and rewards is not None:  # Ortama ait algoritma sonuçları varsa
            # Ödülleri düzgünleştirme (görselleştirme için)
            smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
            plt.plot(smoothed_rewards, label=algo)  # Algoritmayı çiz
    plt.xlabel('Bölümler')  # X ekseni etiketi
    plt.ylabel('Ödüller')  # Y ekseni etiketi
    plt.legend()  # Grafiğe açıklama ekle
    plt.title(f'{env_name} için Algoritma Karşılaştırması')  # Grafik başlığı

    # Ortama özel grafik ölçeği ayarları
    if env_name == "MountainCarContinuous-v0":
        plt.ylim(-1, 0)  # MountainCarContinuous için ölçek
    elif env_name == "Acrobot-v1":
        plt.ylim(-500, 0)  # Acrobot için ölçek
    plt.show()  # Grafiği göster

# Ana fonksiyon
if __name__ == "__main__":
    results = {}  # Sonuçları saklamak için bir sözlük

    print("PPO ile Acrobot-v1 ortamında eğitim başlıyor")
    results["PPO-Acrobot-v1"] = train_ppo_acrobot()  # Acrobot üzerinde PPO eğitimi

    print("DQL ile Acrobot-v1 ortamında eğitim başlıyor")
    results["DQL-Acrobot-v1"] = train_dql_acrobot()  # Acrobot üzerinde DQL eğitimi

    print("DDPG ile MountainCarContinuous-v0 ortamında eğitim başlıyor")
    results["DDPG-MountainCarContinuous-v0"] = train_ddpg_mountaincar()  # MountainCarContinuous üzerinde DDPG eğitimi

    print("PPO ile MountainCarContinuous-v0 ortamında eğitim başlıyor")
    results["PPO-MountainCarContinuous-v0"] = train_ppo_mountaincar()  # MountainCarContinuous üzerinde PPO eğitimi

    # Her ortam için sonuçları çiz
    for env_name in ["Acrobot-v1", "MountainCarContinuous-v0"]:
        plot_results(results, env_name)  # Sonuçları görselleştir
