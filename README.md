# itc-cv-hackathon-fer
the hackathon - building a model to predict facial expressions--learnt a lot about nn and other techniques to improve the mode

I am planning to learn along and explain all the stuff used for the model.

13/09/2026:
Math topics used:


**1. Basic Algebra (the foundation)**
Before any AI, I needed simple maths. Images are stored as numbers 0-255. `ToTensor:81` divides by 255 to make them 0-1. `Normalize:82` does `(pixel - 0.485)/0.229` for each color - like converting marks to a standard scale so the pretrained network understands them. `weights 1/436` for rare `disgust` vs `1/7215` for `happy` is also just algebra.

**2. Linear Algebra (how images become math)**
An image is a grid of numbers - a matrix. A 48x48 grayscale face is a `48x48` matrix; after `convert(L).convert(RGB):68` and `ToTensor` it becomes a `3x224x224` tensor (3 boxes for Red/Green/Blue). Every `Conv2D` in `ResNet34:115` is just sliding a tiny `3x3` number-grid (filter) over the big grid and multiplying - this finds edges, then textures, then eyes and mouth. Without linear algebra, the computer can't see.

**3. Probability (handling uncertainty and rarity)**
The dataset is unfair: `disgust` is only 436 out of 28,709 (1.52%, 1 in 65) while `happy` is 7,215 (25%). If I showed images randomly, the model would rarely see `disgust`. `RandomHorizontalFlip:76`, `Rotation15:77`, `Mixup Beta(0.2):104` are all probability - they randomly change each face so the same 436 `disgust` faces look new every epoch (1 happy+angry blended with `lam` from a Beta dice). `SEED 42:25` just fixes the dice so results repeat.

**4. Statistics (measuring and fixing imbalance)**
I used mean and variance to normalize images, and counting to fix imbalance: `class_weights = 1/count:99` gives `disgust` 16x more penalty when wrong, so the model cares. `train_test_split stratify:53` keeps `1.52% disgust` same in `Train 24402` and `Val 4307` - like keeping the same ratio in an exam sample. `val_acc = correct/total:140` and `gap = train_acc - val_acc` tells me if I memorize (gap 0.11 = mild memorization).

**5. Information Theory: Entropy, Cross-Entropy, KL Divergence (the loss)**
This is the heart - `CrossEntropyLoss:131`. Think of it as a guessing penalty: if true is `fear`, and I predict `fear 0.9` my penalty is small `-log(0.9)=0.10`; if I predict `fear 0.1` penalty is large `-log(0.1)=2.3`. Entropy is how unsure the true label is (0 for sure). KL Divergence is how far my guess is from truth. `CrossEntropy = Entropy + KL`, so minimizing CrossEntropy *is* minimizing KL gap. `label_smoothing 0.1` softens target `1.0 -> 0.9` so I don't become over-confident on noisy FER labels.

**6. Calculus (how the network learns)**
`loss.backward():161` uses the chain rule from calculus - it calculates the slope `∂loss/∂weight` for every one of 21M weights (how much loss would drop if that weight nudged). Then `optimizer.step():163` nudges each weight down that slope. ResNet's skip `y=F(x)+x` is a calculus trick: it lets the slope flow through 34 layers without vanishing to zero, so deep networks can actually learn.

**7. Optimization (finding the best weights)**
Learning is finding the lowest valley in a huge hilly landscape (all weights). `AdamW lr3e-4 weight_decay:134` is the walker - it takes steps `3e-4` big at first, `CosineAnnealing:136` shrinks steps `3e-4->0` over 25 epochs so it settles precisely. `weight_decay` penalizes huge weights (which mean memorizing quirks). `AMP GradScaler:138` just makes 224px training 2x faster. I save only `best checkpoint:168` (highest `val_acc`) not the last epoch, because last is often overfit.


> In short: Linear Algebra turns faces into numbers, Probability/Statistics handle rare `disgust`, Information Theory scores the guess, Calculus+Optimization nudges 21M numbers via `backward` until `Voting Ensemble w_a*p_a+w_b*p_b:190` averages `ResNet34` (deep shape) and `EffNet-B0` (texture attention) to cancel mistakes -> final model.
