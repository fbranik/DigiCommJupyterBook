from scipy import signal
# Ανατρέξτε στην τεκμηρίωση της βιβλιοθήκης scipy.signal
# https://docs.scipy.org/doc/scipy/reference/signal.html
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

### Μέρος 1 Σχεδιασμός και υλοποίηση φίλτρων
---
Εδώ θα πειραματιστούμε με δύο σήματα: (i) το sonar του παραδείγματος, το οποίο εδώ διαβάζεται από ένα .txt αρχείο 
(έχει προέλθει με εξαγωγή του s από το MATLAB) και (ii) ένα σήμα μουσικής, το violin.wav (σήμα από μουσική βιολιού), 
το οποίο περιέχει υψηλότερες συχνότητες και έχει προέλθει με δειγματοληψία στα Fs_viol=44100 Hz.


#### Σήμα sonar

# Ανάγνωση δειγμάτων σήματος από txt file
with open('sima.txt') as f:
    s = [float(x) for x in f]
s=np.array(s)   
print('μέγεθος σήματος =', s.shape)
Fs=8192

#### Στο πεδίο του χρόνου

t=np.arange(0,len(s))/Fs
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax.plot(t,s)
plt.show()

#### Ακούμε το σήμα 

# Πρέπει να έχουμε εγκατεστημένη τη βιβλιοθήκη sounddevice
import sounddevice as sd
sd.play(20*s,Fs)

#### Φάσμα (spectrum) 

f, Pxx_den = signal.welch(s, Fs, noverlap=128, nperseg=256)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

plt.title('Φάσμα σήματος sonar')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Πυκνότητα φάσματος ισχύος ')

ax.semilogy(f, Pxx_den)

#### Σήμα βιολιού

from scipy import signal
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

f=open('violin.wav', 'rb')
Fs_viol, s_viol = scipy.io.wavfile.read(f)
print('Fs_viol=',Fs_viol, ' number of samples=',len(s_viol))
f.close()

#### Στο πεδίο του χρόνου

tvl=np.arange(0,len(s_viol))/Fs_viol
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax.plot(tvl,s_viol)
plt.show()

# Πρέπει να έχουμε εγκατεστημένη τη βιβλιοθήκη sounddevice
import sounddevice as sd
sd.play(s_viol,Fs_viol)

#### Φάσμα (spectrum) και Φασματόγραμμα  (spectorgram)

f, Pxx_den = signal.welch(s_viol, Fs_viol, nperseg=1024, noverlap=256)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax.semilogy(f, Pxx_den)
plt.ylim([0.5e-2, 1e5])
plt.xlabel('Συχνότητα [Hz]')
plt.ylabel('Πυκνότητα φάσματος ισχύος [V**2/Hz]')
plt.show()

f, tsp, Sxx = signal.spectrogram(s, Fs)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax.pcolormesh(tsp, f, Sxx)
plt.ylabel('Συχνότητα [Hz]')
plt.xlabel('Χρόνος [sec]')
plt.show()

#### Βαθυπερατά φίλτρα

#### Η μέθοδος των παραθύρων


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
#
# Fs=8192
H=np.hstack((np.ones(int(Fs/8)), np.zeros(int(Fs-Fs/4)), np.ones(int(Fs/8))))
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax.stem(H)
plt.show()
# Το γράφημα αυτό αργεί... περιμένετε...

#### _Ορθογωνικό παράθυρο (απλή περικοπή της h)_

h=np.real(np.fft.ifft(H));
middle=int(len(h)/2)
h=np.hstack((h[middle:],h[:middle]))
h32=h[middle-16:middle+16]
h128=h[middle-64:middle+64]
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax.stem(h32)
plt.show()

# Σχεδίαση απόκρισης συχνότητας (πλάτους)
#  ΜΠΟΡΕΙ ΝΑ ΓΙΝΕΙ ΣΥΝΑΡΤΗΣΗ !!!

freq,resp32 = signal.freqz(h32);


fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

plt.title('Απόκριση συχνότητας βαθυπερατού φίλτρου\n (με ορθογωνικά παράθυρα)')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Κέρδος')
ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp32), 'b-')
freq,resp128 = signal.freqz(h128);
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp128), 'g-')

plt.show()

#### _Παράθυρα Hamming και Kaiser_

h64=h[middle-32:middle+32]
freq,resp64 = signal.freqz(h64);
w_hamming=signal.hamming(len(h64))
h64_hamming = np.multiply(h64,w_hamming)
w_kaiser=signal.kaiser(len(h64),5)
h64_kaiser = np.multiply(h64,w_kaiser)

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)


plt.title('Απόκριση συχνότητας βαθυπερατού φίλτρου\n παράθυρα hamming (πράσινο) και kaiser (κόκκινο)\n(το αντίστοιχο ορθογωνικό σε μπλε)')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Κέρδος')

ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp64), 'b-')
freq,resp64_hamming = signal.freqz(h64_hamming);
ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp64_hamming), 'g-')
freq,resp64_kaiser = signal.freqz(h64_kaiser);
ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp64_kaiser), 'r-')

plt.show()

#### Φίλτρα ισοϋψών κυματώσεων

lpass = signal.remez(64, [0, 1000, 1300, Fs/2], [1, 0], fs=Fs)

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

plt.title('Απόκριση συχνότητας βαθυπερατού φίλτρου equirriple (πράσινο) \n(το αντίστοιχο με ορθογωνικό παραθυρο σε μπλε)')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Κέρδος')

ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp64), 'b-')
freq,resp_pm = signal.freqz(lpass);
ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp_pm), 'g-')

plt.show()

#### Εφαρμογή του φίλτρου

s_pm = signal.convolve(s,lpass,mode='same')/sum(lpass)

f, Pxx_den = signal.welch(s_pm, Fs, noverlap=128, nperseg=256)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

plt.title('Φάσμα φιλτραρισμένου σήματος sonar')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Πυκνότητα φάσματος ισχύος')
plt.ylim((1e-15,1e-7))

ax.semilogy(f, Pxx_den)
sd.play(s_pm,Fs)



#### Ζωνοπερατά φίλτρα

#### _Με αναλυτικό υπολογισμό της κρουστικής απόκρισης και παράθυρο_ 

# Με αναλυτικό υπολογισμό της κρουστικής απόκρισης και παράθυρο kaiser
f1=800; f2=1600;  
Ts=1/Fs;
f2m1=(f2-f1); f2p1=(f2+f1)/2; N=256
t=np.arange(-(N-1),N-1,2)*Ts/2
hbp=2/Fs*np.divide(np.multiply(np.cos(2*np.pi*f2p1*t),np.sin(np.pi*f2m1*t))/np.pi,t);
hbpw=np.multiply(hbp,signal.kaiser(len(hbp),5));

s_bp=signal.convolve(s,hbp,'same');

f, Pxx_den = signal.welch(s_bp, Fs, noverlap=128, nperseg=256)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

plt.title('Φάσμα ζωνοπερατού σήματος sonar')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Πυκνότητα φάσματος ισχύος')
plt.ylim((1e-15,1e-7))

ax.semilogy(f, Pxx_den)
sd.play(20*s_bp,Fs)

#### _Ζωνοπερατό ισουψών κυματώσεων_ 

bpass = signal.remez(128, [0, f1*0.9, f1*1.1, f2*0.95, f2*1.05, Fs/2], [0, 1, 0], fs=Fs)

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)


plt.title('Απόκριση συχνότητας ζωνοπερατού φίλτρου equirriple')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Κέρδος')

freq,resp_pm = signal.freqz(bpass);
ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp_pm), 'g-')

plt.show()

#### Ζωνοπερατό φίλτρο με ζώνες διέλευσης (750 Hz, 950 Hz) και (3000 Hz, 3500 Hz)

#### _Με αναλυτικό υπολογισμό της κρουστικής απόκρισης και παράθυρο_ 

f1=750 
f2=950
f3=3000
f4=3500
Ts=1/Fs
f2m1=(f2-f1)
f2p1=(f2+f1)/2 
f4m3=(f4-f3)
f4p3=(f4+f3)/2
N=256
t=np.arange(-(N-1),N-1,2)*Ts/2;
band1=2/Fs*np.divide(np.multiply(np.cos(2*np.pi*f2p1*t),np.sin(np.pi*f2m1*t))/np.pi,t)
band2=2/Fs*np.divide(np.multiply(np.cos(2*np.pi*f4p3*t),np.sin(np.pi*f4m3*t))/np.pi,t)
hbp2=band1+band2
hbpw2=np.multiply(hbp2,signal.kaiser(len(hbp2),5));
s_bp2=signal.convolve(s,hbpw2,'same');

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)


plt.title('Απόκριση συχνότητας ζωνοπερατού φίλτρου με ζώνες διέλευσης (750 Hz, 950 Hz) και (3000 Hz, 3500 Hz) και παράθυρο kaiser')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Κέρδος')

freq,resp_pm = signal.freqz(hbpw2);
ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp_pm), 'g-')

plt.show()

f, Pxx_den = signal.welch(s_bp2, Fs, noverlap=128, nperseg=256)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)


plt.title('Φάσμα ζωνοπερατού σήματος sonar (2 ζώνες διέλευσης)')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Πυκνότητα φάσματος ισχύος')
plt.ylim((1e-15,1e-7))

ax.semilogy(f, Pxx_den)
sd.play(20*s_bp2,Fs)

#### _Εφαρμογή φίλτρου στο σήμα violin_ 

s_viol_bp2=signal.convolve(s_viol,hbpw2,'same');

f, Pxx_den = signal.welch(s_viol_bp2, Fs_viol, noverlap=128, nperseg=256)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)


plt.title('Φάσμα ζωνοπερατού σήματος violin (2 ζώνες διέλευσης)')
plt.grid(alpha=0.25)
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Πυκνότητα φάσματος ισχύος')
plt.ylim((1e-15,1e-7))

ax.semilogy(f, Pxx_den)
sd.play(20*s_viol_bp2,Fs_viol)



#### _Ζωνοπερατό ισουψών κυματώσεων με ζώνες διέλευσης (750 Hz, 950 Hz) και (3000 Hz, 3500 Hz)_ 

bpass2 = signal.remez(128, [0, f1*0.8, f1, f2, 1.15*f2, f3*0.95, f3, f4, f4*1.03, Fs/2],
                      [0, 1, 0, 1, 0], fs=Fs)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)


plt.title('Απόκριση συχνότητας ζωνοπερατού φίλτρου με ζώνες διέλευσης (750 Hz, 950 Hz) και (3000 Hz, 3500 Hz) με χρήση της signal.remez')
plt.grid()
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Κέρδος')
freq,resp_pm = signal.freqz(bpass2)

ax.semilogy(0.5*Fs*freq/np.pi, np.abs(resp_pm), 'g-')


s_bpass2 = signal.convolve(s,bpass2,'same')

