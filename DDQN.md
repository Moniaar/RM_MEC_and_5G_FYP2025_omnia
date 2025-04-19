## Important notes based on the DDQN algorithm in our paper:

## We need to do the following:
1. A function that splits the data from the devices, collect 1MB from the dataset
2. The 2 lines that downloads the datasets, which one will we use 1,000 or 5,000
3. function to distribute the dataset on the devices
4. function to select 1MB from the devices that are included in the training


### Line 5:
**"Select an action *a* randomly with probability *ε* or *a = arg max_{a∈A} Q(s_k, a; θ)* with probability *(1 − ε)*."**

This line describes a decision-making process commonly used in reinforcement learning, specifically in the context of an **ε-greedy strategy**. Here’s what it means step-by-step:

1. **Purpose**: The algorithm is deciding which action (*a*) to take in the current state (*s_k*) during iteration *k*. It balances **exploration** (trying new actions randomly) and **exploitation** (choosing the best-known action based on current knowledge).

2. **Two Options**:
   - **Random Action**: With probability *ε* (a small value between 0 and 1), the action *a* is chosen randomly. This encourages exploration of the environment.
   - **Greedy Action**: With probability *(1 − ε)*, the action *a* is chosen as the one that maximizes the action-value function *Q(s_k, a; θ)*. This is the exploitation step, where the algorithm picks the action it currently believes is the best.

3. **Key Symbols**:
   - *a*: The action to be selected.
   - *ε* (epsilon): A parameter (typically small, e.g., 0.1) that controls the trade-off between exploration and exploitation. It represents the probability of choosing a random action.
   - *arg max_{a∈A}*: This mathematical notation means "the action *a* from the set of possible actions *A* that maximizes the following function." In this case, it’s finding the action that gives the highest value of *Q*.
   - *Q(s_k, a; θ)*: The action-value function, which estimates the expected future reward for taking action *a* in state *s_k*, given the current model parameters *θ*. Here:
     - *s_k*: The state at iteration *k*.
     - *a*: The action being evaluated.
     - *θ* (theta): The weights or parameters of the model (e.g., a neural network) that define the *Q*-function.
   - *(1 − ε)*: The probability of choosing the greedy (optimal) action instead of a random one.

### Explanation of the Symbol *ε* (Epsilon):
- *ε* is a Greek letter commonly used in reinforcement learning to denote the exploration rate.
- It determines how often the algorithm explores randomly versus exploiting its current knowledge.
- For example, if *ε = 0.1*, there’s a 10% chance of picking a random action and a 90% chance of picking the action with the highest *Q*-value.

### What Happens in Line 5:
The algorithm flips a biased coin:
- If the result falls within the *ε* probability (e.g., 10%), it picks a random action *a* from the set of possible actions *A*.
- Otherwise, with probability *(1 − ε)* (e.g., 90%), it computes the *Q*-value for all possible actions in the current state *s_k* using the model parameters *θ*, and selects the action *a* that gives the highest *Q*-value.

### In Simple Terms:
Line 5 is like deciding whether to "try something new" (random action) or "go with what you know works best" (maximize *Q*). The symbol *ε* controls how often you take a chance versus playing it safe.

Mini-batch size means:
Taking samples from the memory to train the model

Batch size means: taking samples from the pytorch to train the model


What is Adam?
Adam (short for Adaptive Moment Estimation) is an algorithm that adjusts the learning rate for each parameter automatically. It combines the benefits of two other methods:

Momentum (uses past gradients to smooth out updates)

RMSProp (scales learning rate based on past gradient magnitudes)

Below is the translated comparison table in Arabic, covering all aspects from the previous comparison of the `DeviceSelectionEnv` class (first code) and the combination of `EdgeDevice`, `FiveGNetwork`, `MECServer`, and `DDQN` classes (second code). The table includes the functionality, inputs, outputs, and key differences for each aspect, as requested.

| **الجانب** | **الكود الأول: DeviceSelectionEnv** | **الكود الثاني: EdgeDevice, FiveGNetwork, MECServer, DDQN** | **الاختلافات الرئيسية** |
|-------------|--------------------------------------|-------------------------------------------------------------|---------------------------|
| **نظرة عامة على الفئة** | `DeviceSelectionEnv`: بيئة تعلم معزز (RL) لاختيار الأجهزة، تدير الحالة (التأخير، النطاق الترددي، الطاقة)، الإجراءات، والمكافآت. | `EdgeDevice`: نمذجة الأجهزة الفردية مع التدريب المحلي، الطاقة، والنطاق الترددي. `FiveGNetwork`: محاكاة ديناميكيات شبكة الجيل الخامس. `MECServer`: تنسيق التدريب والتجميع. `DDQN`: إدارة اختيار الأجهزة باستخدام التعلم المعزز. | الكود الأول يستخدم فئة واحدة للبيئة، بينما الكود الثاني يقسم المسؤوليات عبر فئات متعددة لتحقيق التجزئة. |
| **التهيئة (`__init__`)** | **DeviceSelectionEnv**: تهيئ عدد الأجهزة، الحد الأقصى للطاقة (\( E_{\text{MAX}} \))، الحالة، وفضاء الإجراءات. المدخلات: `num_devices`. | **EdgeDevice**: تهيئ معرف الجهاز، تردد المعالج، الطاقة، النطاق الترددي، ونموذج CNN. المدخلات: `id`, `cpu_freq`, `energy`, `bandwidth`, `fiveg_network`. **FiveGNetwork**: تحديد النطاق الترددي الأساسي، التأخير، والسعة. المدخلات: `base_bandwidth`, `latency`, `capacity`. **MECServer**: إنشاء الأجهزة وتهيئة نموذج CNN العالمي. المدخلات: `num_devices`. **DDQN**: إعداد شبكة عصبية للتعلم المعزز. المدخلات: `state_size`, `action_size`. | الكود الأول يحتوي على تهيئة مركزية أبسط. الكود الثاني يوزع التهيئة عبر فئات، مع `FiveGNetwork` تضيف نمذجة شبكة صريحة. |
| **توليد الحالة (`generate_state`)** | يولد مصفوفة حالة (\( N \times 3 \)) مع التأخير (\( U[0, L_{\text{MAX}}] \))، النطاق الترددي (\( U[0, 2] \))، والطاقة (مبدئيًا \( E_{\text{MAX}} \)، يتم تحديثها لاحقًا). المدخلات: `initial` (منطقي). المخرجات: مصفوفة الحالة. | **EdgeDevice (`report_resources`)**: يعيد حالة الجهاز (تردد المعالج، الطاقة، النطاق الترددي). **MECServer**: يجمع الحالات من الأجهزة عبر `report_resources`. لا يوجد `generate_state` صريح، لكن الحالة تُبنى في `select_devices_ddqn`. المخرجات: مصفوفة حالة مسطحة. | الكود الأول يركز توليد الحالة في البيئة. الكود الثاني يبني الحالة بشكل مؤقت من تقارير الأجهزة، ويفتقر إلى طريقة مخصصة. |
| **معالجة الإجراءات/الخطوة (`step`)** | يعالج الإجراء (متجه ثنائي)، يحسب التأخير (\( T_{\text{local}} + T_{\text{trans}} \))، استهلاك الطاقة (\( B_k \))، والمكافأة (\( R = \alpha_n \cdot \frac{m}{n} - \alpha_e \cdot \frac{E}{E_{\text{max_total}}} - \alpha_l \cdot \frac{L}{L_{\text{max}}} \)). يحدث الطاقة و \( E_{\text{MAX}} \). المدخلات: `action`. المخرجات: الحالة التالية، المكافأة، الانتهاء. | **EdgeDevice (`train_local_model`)**: يدرب النموذج المحلي، يحسب \( T_{\text{local}} \)، \( T_{\text{trans}} \)، \( B_k \)، ويتحقق من الطاقة. المدخلات: `epochs`, `energy_rate`. المخرجات: الأوزان، \( T_{\text{local}} \)، \( T_{\text{trans}} \)، \( B_k \). **MECServer (`simulate_training_round`)**: ينسق التدريب للأجهزة المختارة، يجمع الأوزان، يحسب أقصى تأخير. المدخلات: `selected_devices`, `epochs`. المخرجات: أقصى تأخير. **DDQN (`act`)**: يختار إجراء (فهرس الجهاز). المدخلات: `state`. المخرجات: الإجراء. | الكود الأول يدمج كل المنطق (التأخير، الطاقة، المكافأة) في طريقة واحدة. الكود الثاني يقسم المنطق عبر الفئات، مع `MECServer` يدير التنسيق و `EdgeDevice` يجري الحسابات. الكود الثاني يتضمن فحص كفاية الطاقة. |
| **إعادة التعيين (`reset`)** | يعيد تعيين \( E_{\text{MAX}} \)، إجمالي الطاقة، والحالة إلى القيم الأولية. المدخلات: لا شيء. المخرجات: الحالة الأولية. | لا توجد طريقة إعادة تعيين صريحة. **MECServer**: يعيد التعيين ضمنيًا عبر تهيئة الجولة الجديدة. **EdgeDevice**: الطاقة تُحدث في `train_local_model`، لا إعادة تعيين كاملة. | الكود الأول يحتوي على إعادة تعيين واضحة للحلقات المعززة. الكود الثاني يفتقر إلى إعادة التعيين، مفترضًا التشغيل المستمر عبر الجولات. |
| **حساب الطاقة** | \( B_k = f_k^2 \cdot \tau \cdot \mu_{\text{bits}} \cdot G \)، \( C_k \sim \text{Poisson}(\lambda) \)، \( e_k' = \max(e_k - B_k + C_k, 0) \). يحدث \( E_{\text{MAX}} \). | **EdgeDevice**: نفس صيغ \( B_k \)، \( C_k \)، وتحديث الطاقة. يتضمن فحص كفاية الطاقة (\( e_k < B_k \)). | صيغ الطاقة متطابقة، لكن الكود الثاني يضيف فحصًا لتخطي التدريب إذا كانت الطاقة غير كافية. |
| **حساب التأخير** | \( T_{\text{local}} = \frac{\mu_{\text{bits}} \cdot G}{f_k} \)، \( T_{\text{trans}} = \frac{D}{r_k} \)، \( L = \max \{ T_{\text{local}} + T_{\text{trans}} \} \). | **EdgeDevice**: نفس \( T_{\text{local}} \)، \( T_{\text{trans}} \). **MECServer**: يحسب \( \max \{ T_{\text{local}} + T_{\text{trans}} \} \). | صيغ التأخير وتطبيق الحد الأقصى متطابقة. |
| **حساب المكافأة** | \( R = \alpha_n \cdot \frac{m}{n} - \alpha_e \cdot \frac{E}{E_{\text{max_total}}} - \alpha_l \cdot \frac{L}{L_{\text{max}}} \). | **الحلقة الرئيسية**: \( R = \text{accuracy} - \frac{\text{delay}}{1000} \). | الكود الأول يستخدم مكاف Alphas يوازن بين عدد الأجهزة، الطاقة، والتأخير. الكود الثاني يستخدم الدقة والتأخير، أبسط لكنه أقل توافقًا مع قيود الموارد. |
| **نمذجة الشبكة** | ضمنية في الحالة (النطاق الترددي \( U[0, 2] \)). | **FiveGNetwork**: نمذجة شبكة الجيل الخامس صريحة مع تخصيص النطاق الترددي الديناميكي (\( \text{base_bandwidth} \cdot (1 - 0.05 \cdot \text{connected_devices}) \)). | الكود الثاني ينمذج ديناميكيات الشبكة بشكل صريح، مما يضيف واقعية ولكن يزيد التعقيد. |
| **اختيار الأجهزة** | **DDQNAgent (`select_action`)**: يستخدم DDQN قائم على CNN، يختار أجهزة متعددة عبر متجه ثنائي. | **DDQN (`act`)**: يستخدم DDQN قائم على MLP أبسط، يختار جهازًا واحدًا لكل استدعاء، يتكرر لاختيار أجهزة متعددة. | DDQN الكود الأول أكثر تعقيدًا (CNN) ويختار أجهزة متعددة دفعة واحدة. DDQN الكود الثاني أبسط (MLP) لكنه أقل كفاءة لاختيار أجهزة متعددة. |
| **تنسيق التدريب** | **الحلقة الرئيسية**: تكرار الحلقات، اختيار الأجهزة، تدريب النماذج المحلية، تجميع الأوزان. | **MECServer (`simulate_training_round`)**: ينسق التدريب، يجمع الأوزان. **الحلقة الرئيسية**: تدير الجولات وتحديثات DDQN. | تنسيق مشابه، لكن `MECServer` الكود الثاني يوفر فصلًا أوضح للمهام (مثل توزيع البيانات، التجميع). |
| **المعاملات** | \( G = 7000 \)، \( \tau = 10^{-28} \)، \( \mu_{\text{bits}} = 8 \cdot 10^6 \)، \( D = 160 \cdot 10^6 \)، \( \lambda = 1 \)، \( E_{\text{MAX}} = 5 \). | نفسها، باستثناء الطاقة الأولية (\( 5000 + \text{cpu_freq} \cdot 1000 \)) ومعاملات الشبكة (\( \text{base_bandwidth} = 100 \)، إلخ). | متطابقة في الغالب، لكن تهيئة الطاقة في الكود الثاني ديناميكية، ومعاملات الشبكة تضيف تعقيدًا. |

How is CIFAR-10 Dataset used here?:
Each class contains an equal distribution, boasting 6,000 images. From the total image count, 50,000 are designated for training while the remaining 10,000 are set aside for testing.
