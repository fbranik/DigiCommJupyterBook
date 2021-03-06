from scipy import signal
# Ανατρέξτε στην τεκμηρίωση της βιβλιοθήκης scipy.signal
# https://docs.scipy.org/doc/scipy/reference/signal.html
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import warnings
warnings.filterwarnings('ignore')


### Μέρος 2:  Δειγματοληψία - Ψηφιοποίηση
---
Τα πρωτογενή σήματα είναι κυρίως αναλογικά (συνεχούς χρόνου). Για να τα παραστήσουμε και επεξεργαστούμε στον υπολογιστή μας (ή άλλη ψηφιακή μηχανή) θα πρέπει πρώτα να τα ψηφιοποιήσουμε. Υποθέστε ένα σήμα συνεχούς χρόνου $x(t)$ με μετασχηματισμό Fourier (Continuous Time Fourier Transform – CTFT):
 $X(f)=\int_{-\infty}^{\infty} x(t)e^{-j2\pi ft} dt$
![1.1](images/1_1.png)

Λαμβάνοντας δείγματα του $x(t)$ με ρυθμό $f_s=1/T_s$ παράγεται σήμα διακριτού χρόνου $x(nT_s)$. Μαθηματικά το αναπαριστάνουμε ως σειρά συναρτήσεων δέλτα  

$$x_\delta (t)=\sum_{n=-\infty}^{\infty}x(nT_s)\delta(t-nT_s)=x(t\sum_{n=-\infty}^{\infty}\delta(t-nT_s)$$  

με μετασχηματισμό Fourier

$$X_\delta (f)=\sum_{n=-\infty}^{\infty}x(nT_s)e^{-j2\pi fnT_s}=X(f)*1/T_s\sum_{n=-\infty}^{\infty}\delta(f-k/T_s)=1/T_s\sum_{n=-\infty}^{\infty}X(f-k/T_s)$$

που είναι περιοδική συνάρτηση.<br />
![1.2](images/1.2.png)

Για βαθυπερατά σήματα $x(t)$ εύρους ζώνης W, με την υπόθεση ότι ο ρυθμός δειγματοληψίας $fs ≥
2W$, ισχύει ότι $X(f) = T_s X_\delta(f)$, $0 ≤ f ≤ W$, δηλαδή, το σήμα $X(f)$ προκύπτει μετά από διάβαση του
δειγματοληπτημένου $x_\delta(t)$ μέσω ιδανικού βαθυπερατού φίλτρου κέρδους $T_s$. Από το προηγούμενο
σχήμα γίνεται φανερό ότι εάν η δειγματοληψία γίνει με συχνότητα μικρότερη του διπλασίου της
ανώτερης συχνότητας $W$ του σήματος (υποδειγμάτιση – undersampling), τότε εμφανίζονται στην
περιοχή συχνοτήτων του σήματος «είδωλα» φάσματος από ανώτερες συχνότητες που δεν
επιτρέπουν την ακριβή αποκατάσταση του αρχικού σήματος συνεχούς χρόνου. Το φαινόμενο αυτό
ονομάζεται __αναδίπλωση__ ή __επικάλυψη__ (aliasing), το δε σφάλμα κατά την αποκατάσταση του
αρχικού σήματος αποκαλείται σφάλμα αναδίπλωσης (aliasing error).
Η δειγματοληψία στο πεδίο του χρόνου αποτελεί τη βάση για τον ορισμό του μετασχηματισμού
Fourier διακριτού χρόνου (Discrete Time Fourier Transform – DTFT). Για μια σειρά διακριτών
αριθμών $x[n]$, ο μετασχηματισμός Fourier διακριτού χρόνου ορίζεται ως:
$X_\delta (\phi)\triangleq\sum_{n=-\infty}^{\infty}x[n]e^{-j2\pi n\phi}$

O DTFT είναι περιοδική συνάρτηση με περίοδο $1$, επομένως, αρκεί ο υπολογισμός του στο
διάστημα συχνοτήτων $[0,1]$ ή ισοδύναμα $[-½,½]$. Να σημειωθεί ότι ο DTFT, παρότι προκύπτει από
μια σειρά διακριτών αριθμών $x[n]$, είναι συνεχής συνάρτηση της μεταβλητής $\phi$ όπως παραστατικά
φαίνεται στο επόμενο σχήμα.
![1.4](images/1.4.png)

Με τη σειρά των διακριτών αριθμών να προκύπτει ως αποτέλεσμα δειγματοληψίας, $x[n]=x(nT_s)$, ο
DTFT και ο μετασχηματισμός Fourier $X_\delta(f)$ του δειγματοληπτημένου σήματος συνδέονται μέσω
της αντιστοιχίας $\phi ↔ f/f_s$. Η συνήθης πρακτική είναι να παριστάνουμε τον λόγο $f/f_s$ ως
κανονικοποιημένη συχνότητα $\phi$ ($f_D$, στις σημειώσεις σας) και οι πραγματικές συχνότητες να
προκύπτουν ως πολλαπλάσιά της (συνήθως κλασματικά). Για τη σύνδεση του DTFT με τον μετασχηματισμό Fourier $X(f)$ του σήματος πρέπει επιπλέον να γίνει αναγωγή στην περίοδο δειγματοληψίας με πολλαπλασιασμό επί $T_s$ (ή διαίρεση με $f_s$).
Κατ΄ αναλογία με τη δειγματοληψία σημάτων στο χρόνο μπορούμε να κάνουμε δειγματοληψία στο
πεδίο της συχνότητας λαμβάνοντας διακριτές τιμές $X(kf_o)$ του μετασχηματισμού Fourier που
αντιστοιχούν σε ανάλυση συχνότητας $f_o=1/T_o$. Αυτό ισοδυναμεί με περιοδική επανάληψη του
σήματος συνεχούς χρόνου $x(t)$ κάθε $Τ_ο$, αφού το περιοδικό σήμα
$x_p (t)=\sum_{n=-\infty}^{\infty}x(t-nT_o)$

έχει μετασχηματισμό Fourier

$$X(f)\sum_{n=-\infty}^{\infty}e^{-j2\pi f n T_o}= X(f)\frac{1}{T_o}\sum_{k=-\infty}^{\infty}\delta(f-\frac{k}{T_o})=\frac{1}{T_o}\sum_{k=-\infty}^{\infty}X(\frac{k}{T_o})\delta(f-\frac{k}{T_o})$$



Επομένως, $X[k] = X(kf_o)/Τ_o$ είναι οι συντελεστές του αναπτύγματος σε σειρά Fourier.του περιοδικού
σήματος $x_p(t)$. Προφανώς, για σήματα $x(t)$ πεπερασμένης διάρκειας, όπου $x(t)=0$ για $|t| ≥ T$, με την
υπόθεση ότι η περίοδος $T_o ≥ 2T$, ισχύει ότι $x(t) = x_p(t)$ για $|t| ≤ T$.
Στην πράξη, τα σήματα έχουν πολύ μεγάλη διάρκεια για να μπορέσουμε να τα αναλύσουμε στην
ολότητά τους. Έτσι εφαρμόζουμε ένα ορθογωνικό χρονικό παράθυρο, ώστε να διατηρήσουμε μόνο
το πιο σημαντικό τους μέρος για το διάστημα παρατήρησης και $x(t)= 0$, αλλού. Κατά τον
υπολογισμό του DTFT $X_d(\phi)$ ενός τέτοιου ακρωτηριασμένου σήματος, αντί του απείρου
αθροίσματος, περιοριζόμαστε σε μια πεπερασμένου μήκους $L$ σειρά αριθμών $x[n]$, οπότε



$$X_d (\phi)=\sum_{n=0}^{L-1}x[n]e^{-j2\pi n\phi}$$



H δειγματοληψία του $X_d(\phi)$ στο πεδίο συχνότητας σε $Ν$ ισαπέχουσες κανονικοποιημένες συχνότητες $0$, $1/Ν$, $2/Ν$, $…$, $(Ν-1)/Ν$, δίνει

$$X[k]=X_d (\frac{k}{N})=\sum_{n=0}^{N-1}x[n]e^{-j2\pi n\frac{k}{N}}, 0\leq k \leq N-1$$

όπου, εάν $N≥L$, θέτουμε $x[n]=0$ για $n≥L$. Η τελευταία σχέση αναγνωρίζεται ως ο διακριτός μετασχηματισμός Fourier (Discrete Fourier Transform – DFT), ο οποίος για μια πεπερασμένη σειρά $xn$, $n=0$, $1$, $…$, $N-1$, ορίζεται ως:

$$X_k\triangleq \sum_{n=0}^{N-1}x_n e^{-j2\pi n\frac{k}{N}}, 0\leq k \leq N-1$$

και ο αντίστροφός του είναι

$$x_n= \frac{1}{N} \sum_{k=0}^{N-1}X_k e^{-j2\pi n\frac{k}{N}}, 0\leq n \leq N-1$$

Η $X_d(\phi)$ ως DTFT είναι περιοδική συνάρτηση και εάν η αρχική σειρά xn ήταν περιοδική (και δεν
εφαρμόζαμε το παράθυρο), τότε η $X_d(\phi)$ θα ήταν μηδέν παντού εκτός των σημείων της
δειγματοληψίας $k/Ν$. Δηλαδή, εάν θεωρήσουμε μια πεπερασμένου μήκους σειρά αριθμών που
επαναλαμβάνεται περιοδικά, o διακριτού χρόνου μετασχηματισμός Fourier της (DTFT) είναι και
αυτός περιοδικός και διακριτός. Επιπλέον, ο DFT και ο αντίστροφός του IDFT, εάν δεν
περιορίζαμε τους δείκτες $n$ και $k$ μεταξύ $0$ και $N-1$, θα ήταν περιοδικές συναρτήσεις. Άρα η
πεπερασμένη σειρά xn μπορεί να θεωρηθεί ως ένα περιοδικό σήμα διακριτού χρόνου ιδωμένο μόνο
κατά τη διάρκεια μιας περιόδου και ο DFT, η σειρά $X_k$, ως τα δείγματα με ανάλυση $1/Ν$ του DTFT
$X_d(\phi)$ στο πεδίο κανονικοποιημένων συχνοτήτων $[0,1]$, όπως φαίνεται στο επόμενο σχήμα.

![1.4](images/1.4.png)

#### Φασματική Ανάλυση
Για τον υπολογισμό της ενέργειας ή ισχύος της κυματομορφής $x(t)$, ανάλογα με την περίπτωση
σήματος, ισχύει

$$E_x= \int_{-\infty}^{\infty} x^2(t) dt = \int_{-\infty}^{\infty} |X(f)|^2 df$$

$$P_x= \lim_{T\to\infty} \frac{1}{T} \int_{-\frac{T}{2}}^{\frac{T}{2}} x^2(t) dt = \int_{-\infty}^{\infty} S_X(f) df$$

όπου για σήματα ισχύος $S_Χ(f)$ είναι η πυκνότητα φάσματος ισχύος (Power Spectral Density – PSD)
της $x(t)$. Για σήματα διακριτού χρόνου που προκύπτουν από δειγματοληψία της $x(t)$ με περίοδο $T_s$,
οι αντίστοιχες σχέσεις υπολογισμό της ενέργειας ή ισχύος γίνονται

$$E_x= T_s\sum_{n=-\infty}^{\infty} x^2[n]$$

$$P_x= \lim_{N\to\infty} \frac{1}{2N+1} \sum_{n=-N}^{N} x^2[n]$$

Ένας απλός τρόπος να εκτιμηθεί η πυκνότητα φάσματος ισχύος της κυματομορφής $x(t)$ είναι να
ληφθεί ο DTFT των δειγμάτων του σήματος και μετά να υψωθεί στο τετράγωνο το μέτρο του
αποτελέσματος. Αυτός ο εκτιμητής αποκαλείται περιοδόγραμμα (periodogram). Το περιοδόγραμμα
ενός πεπερασμένου μήκους $L$ σήματος $x[n]$ ορίζεται ως

$$P_{xx}(f)\triangleq \frac{|X_d(\frac{f}{f_s})| ^2}{f_sL}$$

όπου $X_d(\phi)$ o DTFT του σήματος. Με το μήκος $L$ να τείνει στο άπειρο, το περιοδόγραμμα $P_{xx}(f)$
τείνει στην πυκνότητα φάσματος ισχύος $S_Χ(f)$. Ο υπολογισμός του περιοδογράμματος σε
πεπερασμένο πλήθος συχνοτήτων $kf_s/Ν$, $k=0$, $1$, $…$ , $Ν$ δίνει

$$P_{xx}[k]=\frac{|X_k| ^2}{f_sL} , k=0, 1, ...,N-1$$

όπου $X_k$ και ο DFT της πεπερασμένου μήκους $L$ σειράς δειγμάτων του σήματος. Η ισχύς του
σήματος είναι τότε

$$P_{X}=\frac{1}{f_sL} \sum_{k=0}^{N-1} |X_k|^2f_o =\frac{1}{NL} \sum_{k=0}^{N-1} |X_k|^2 = \frac{1}{L}\sum_{n=0}^{L-1} |x_n|^2$$

όπου η τελευταία ισότητα προκύπτει από το θεώρημα Parseval, που για την περίπτωση του DFT
εκφράζεται ως:

$$ \sum_{n=0}^{N-1} |x_n|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X_k|^2$$

Στην ειδική περίπτωση περιοδικών σημάτων έχουμε

$$S_X(f)=\sum_{k=-\infty}^{\infty} |X[k]|^2\delta(f-\frac{k}{T_o})$$

$$P_X(f)=\sum_{k=-\infty}^{\infty} |X[k]|^2$$

όπου $X[k]$ οι συντελεστές του αναπτύγματος σε σειρά Fourier και $T_o$ η περίοδος του σήματος.