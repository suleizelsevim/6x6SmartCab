Bu repo, klasik Taxi problemine biraz daha zorluk katarak oluşturduğum 6x6 Custom Environment 
üzerinde çalışan bir Reinforcement Learning uygulamasını içeriyor.

Standart Gymnasium Taxi ortamı yerine kendi belirlediğim duvarlar, yasaklı bölgeler ve 
ödül mekanizmalarıyla sıfırdan bir environment tasarladım. Amaç, ajanın (taksinin) yolcuyu 
en kısa yoldan alıp hedefe engellere takılmadan bırakmasını sağlamak.

## Nasıl Çalıştırılır?

>Repoyu klonladıktan sonra gerekli paketleri kurun:

```bash
pip install -r requirements.txt
```

Ardından kodu çalıştırın

```bash
python 6x6SmartCab.py
```

## Ortam Tasarımı

6x6 grid yapısında şehir ortamı tasarlandı.
Yasak hücreler tanımlandı. Bu hücrelere taksi ve yolcu giremez.
```bash
(1, 1), (2, 3), (4, 4)
```
>Bu hücreler animasyonda kırmızı gösterilir.

Ortama duvarlar eklendi. İki hücre arasında geçişi engeller ve animasyonda kalın siyah çizgiyle gösterilir.
```bash
    (0, 4): {"S": True},   # E -> sağa duvar
    (0, 5): {"S": True},   # S -> aşağı duvar
    (1, 4): {"S": True},   # W -> sola duvar
    (1, 5): {"S": True},   # N -> yukarı duvar
    (4, 0): {"E": True},   
    (5, 0): {"E": True},
    (4, 1): {"E": True}, 
    (5, 1): {"E": True},
```

## Aksiyon Uzayı

Ajanın yapabileceği 6 aksiyon vardır.

|           | Aksiyon | Açıklama |
| ------------- | ------------- | -----------|
| 0  | South  | Güneye (aşağı) git |
| 1  | North  | Kuzeye (yukarı) git |
| 2  | East | Doğuya (sağa) git |
| 3  | West | Batıya (sola) git |
| 4  | Pickup | Yolcu alma |
| 5  | Dropoff | Yolcu bırakma |

## Ödül Sistemi

Ajanın davranışlarını yönlendirmek için ödül/ceza sistemi kullanılır:

```bash
step_penalty   = -0.5     # Her adımda küçük ceza
wall_penalty   = -10.0    # Duvara çarma cezası
illegal_penalty = -20.0   # Yanlış pickup/dropoff
pickup_reward  = 5.0      # Yolcuyu almak
success_reward = 50.0     # Başarıyla hedefe bırakmak
shaping        = 0.5      # Mesafe azaldığında ödüllendirme
```
>Her adımda ajanın hedefe uzaklığı hesaplanıyor ve ajan hedefe yaklaştıysa ödül kazanıyor, uzaklaştıysa ceza yiyor.

## Q-Learning

Ajan q-learning algoritmasıyla eğitiliyor.

Parametreler:

|Parametre| Değer | Açıklama |
| ------------- | ------------- | -----------|
| alpha  | 0.05  | Öğrenme oranı |
| gamma  | 0.995  | Gelecek ödüllere verilen önem |
| epsilon  | 1.0 > 0.1 | Lineer azalır |
| episodes  | 150000 | Eğitim sayısı |

>Epsilon ilk başta yüksek tutulur ve ajan keşif (exploration) yapar, düştükçe sömürü (exploitation) aşamasına geçilir.

## Eğitim Süreci

Eğitim sırasında her 5000 adımda log alınır.

```bash
Episode 5000/150000 - epsilon=0.970 - last_total_reward=-11505.00
Episode 10000/150000 - epsilon=0.940 - last_total_reward=-9102.50
Episode 15000/150000 - epsilon=0.910 - last_total_reward=-4406.00
```
Görüldüğü üzere eğitimin başlarında ajanın yediği ceza çok yüksektir.
Zamanla taksi yolcuya daha hızlı ulaşmayı öğrenir.

```bash
Episode 140000/150000 - epsilon=0.160 - last_total_reward=1.50
Episode 145000/150000 - epsilon=0.130 - last_total_reward=35.00
Episode 150000/150000 - epsilon=0.100 - last_total_reward=43.00
```
![]()
>Episode sayısına göre LastTotalReward değerinin değişimi
## Sonuçların değerlendirilmesi

Eğitim sonrası ajan 20 bölümde test edilir.

```bash
=== Evaluation Results ===
Episodes       : 20
Success count  : 20
Success rate   : 1.00
Avg steps      : 11.80
Avg total reward: 43.65
```
Ajan %100 başarı oranına ulaşmıştır.

![](https://github.com/suleizelsevim/6x6SmartCab/blob/master/smartcab_6x6.gif)
