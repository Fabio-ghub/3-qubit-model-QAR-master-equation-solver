import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, qeye, destroy, sigmaz, steadystate, fock_dm

def solve_qar_steadystate(E1, E2, g, p1, p2, p3, T1, T2, T3):
    """
    Risolve la Master Equation del Quantum Absorption Refrigerator per 3 qubit.
    Restituisce la temperatura stazionaria T1_S e il flusso di calore Q1_S.
    """
    E3 = E2 - E1
    
    # Operatori identità e distruzione per qubit singoli
    I = qeye(2)
    sm = destroy(2)
    sp = sm.dag()
    sz = sigmaz()
    
    # Proiettori |1><1| per ogni qubit
    P1 = fock_dm(2, 1)
    
    # Hamiltoniana Libera H0
    H0 = E1 * tensor(P1, I, I) + \
         E2 * tensor(I, P1, I) + \
         E3 * tensor(I, I, P1)
         
    # Hamiltoniana di interazione Hint: g(|101><010| + |010><101|)
    state_101 = tensor(basis(2, 1), basis(2, 0), basis(2, 1))
    state_010 = tensor(basis(2, 0), basis(2, 1), basis(2, 0))
    Hint = g * (state_101 * state_010.dag() + state_010 * state_101.dag())
    
    H = H0 + Hint
    
    # Costruzione degli operatori di collasso di Lindblad per riprodurre esattamente 
    # la mappa di dissipazione: p_i * (tau_i * Tr_i(rho) - rho)
    c_ops = []
    params = [(p1, E1, T1, 0), (p2, E2, T2, 1), (p3, E3, T3, 2)]
    
    for p, E, T, idx in params:
        if p == 0: continue
        
        # Probabilità di ground state r_i
        # Evitiamo divisioni per zero se T -> 0
        T_eff = max(T, 1e-12)
        r = 1.0 / (1.0 + np.exp(-E / T_eff))
        
        # Operatori di innalzamento, abbassamento e dephasing nel sottospazio corretto
        op_m = [I, I, I]
        op_m[idx] = sm
        op_m = tensor(op_m)
        
        op_p = [I, I, I]
        op_p[idx] = sp
        op_p = tensor(op_p)
        
        op_z = [I, I, I]
        op_z[idx] = sz
        op_z = tensor(op_z)
        
        # I tre rate esatti per riprodurre il decadimento generalizzato del paper
        c_ops.append(np.sqrt(p * r) * op_m)
        c_ops.append(np.sqrt(p * (1.0 - r)) * op_p)
        c_ops.append(np.sqrt(p / 2.0) * op_z)
        
    # Calcolo dello stato stazionario
    rho_ss = steadystate(H, c_ops)
    
    # Estrazione delle metriche del qubit 1 (traccia parziale su 2 e 3)
    rho1_ss = rho_ss.ptrace(0)
    r1_s = rho1_ss[0, 0].real
    
    # Calcolo Temperatura Stazionaria T1_S
    if r1_s >= 0.999999: 
        T1_s = 0.0 # Vicino allo zero assoluto
    else:
        T1_s = E1 / np.log(r1_s / (1.0 - r1_s))
        
    # Calcolo Flusso di Calore Q1_S = Tr(H1 * D1(rho_ss))
    # Semplificato esattamente a: p1 * E1 * (r1_s - r1_bath)
    r1_bath = 1.0 / (1.0 + np.exp(-E1 / max(T1, 1e-12)))
    Q1_s = p1 * E1 * (r1_s - r1_bath)
    
    return T1_s, Q1_s

# ==========================================
# GENERAZIONE DEI GRAFICI REPLICATI
# ==========================================

plt.figure(figsize=(14, 10))

# 1. Grafico: T1_S vs T_h (Fig. 1 del paper)
plt.subplot(2, 2, 1)
Th_vals = np.linspace(0.1, 10, 50)
Tc_vals = [2.0, 1.5, 1.0]

for Tc in Tc_vals:
    T1_diffs = []
    for Th in Th_vals:
        # Parametri standard da Fig 1
        T1_s, _ = solve_qar_steadystate(E1=1, E2=5, g=0.01, p1=0.01, p2=0.01, p3=0.01, 
                                        T1=Tc, T2=2.0, T3=Th)
        T1_diffs.append(T1_s - Tc)
    plt.plot(Th_vals, T1_diffs, label=f'$T_c$ = {Tc}')

plt.axhline(0, color='black', linewidth=1)
plt.xlabel('Temperatura del bagno caldo $T_h$')
plt.ylabel('$T_1^S - T_c$')
plt.title('Raffreddamento: Differenza di Temperatura vs $T_h$')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Grafico: Flusso di calore stazionario Q1_S vs T_h (Fig. 2 del paper)
plt.subplot(2, 2, 2)
for Tc in Tc_vals:
    Q1_vals = []
    for Th in Th_vals:
        _, Q1_s = solve_qar_steadystate(E1=1, E2=5, g=0.01, p1=0.01, p2=0.01, p3=0.01, 
                                        T1=Tc, T2=2.0, T3=Th)
        Q1_vals.append(Q1_s)
    plt.plot(Th_vals, Q1_vals, label=f'$T_c$ = {Tc}')

plt.axhline(0, color='black', linewidth=1)
plt.xlabel('Temperatura del bagno caldo $T_h$')
plt.ylabel('Flusso di calore $Q_1^S$')
plt.title('Flusso di calore stazionario vs $T_h$')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Grafico: T1_S vs rate di termalizzazione p2 e p3 (Fig. 3 del paper)
plt.subplot(2, 2, 3)
p_vals = np.linspace(0.001, 1, 100)
T1_s_p2_var = []
T1_s_p3_var = []

for p in p_vals:
    # Vario p2, fisso p3 = 0.05
    T1, _ = solve_qar_steadystate(E1=1, E2=5, g=0.05, p1=0.05, p2=p, p3=0.05, 
                                  T1=1.0, T2=1.0, T3=3.0)
    T1_s_p2_var.append(T1)
    
    # Vario p3, fisso p2 = 0.05
    T1, _ = solve_qar_steadystate(E1=1, E2=5, g=0.05, p1=0.05, p2=0.05, p3=p, 
                                  T1=1.0, T2=1.0, T3=3.0)
    T1_s_p3_var.append(T1)

plt.plot(p_vals, T1_s_p2_var, '--', label='Vario $p_2$ (fisso $p_3=0.05$)')
plt.plot(p_vals, T1_s_p3_var, ':', label='Vario $p_3$ (fisso $p_2=0.05$)')
plt.xlabel('Rate di termalizzazione $p_2$ o $p_3$')
plt.ylabel('Temperatura Stazionaria $T_1^S$')
plt.title("Effetto Zeno Quantistico (Fig. 3)")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Grafico: T1_S vs Energy Gap E2 verso lo zero assoluto (Fig. 4 del paper)
plt.subplot(2, 2, 4)
E2_vals = np.linspace(1.5, 100, 100)
T1_s_E2 = []

for E2 in E2_vals:
    E3 = E2 - 1.0 # Dato che E1 = 1
    # Th scala proporzionalmente a E3 per mantenere E3/Th costante e << 1, 
    # come descritto nel testo per approssimare lo zero assoluto.
    Th = 10*E2
    
    T1, _ = solve_qar_steadystate(E1=1, E2=E2, g=0.05, p1=1e-6, p2=0.3, p3=0.1, 
                                  T1=2.0, T2=2.0, T3=Th)
    T1_s_E2.append(T1)

plt.plot(E2_vals, T1_s_E2, 'r-')
plt.xlabel('Energy gap della "spirale" $E_2$')
plt.ylabel('Temperatura Stazionaria $T_1^S$')
plt.title('Raffreddamento verso lo Zero Assoluto (Fig. 4)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot fridge')
plt.show()

# --- I TUOI PARAMETRI PERSONALIZZATI ---
E1_custom = 1.0     # Energia qubit 1 (da raffreddare)
E2_custom = 5.0     # Energia qubit 2 (motore/spirale)
g_custom  = 0.02    # Forza di interazione tra i qubit
p1_custom = 0.01    # Rate di termalizzazione qubit 1
p2_custom = 0.01    # Rate di termalizzazione qubit 2
p3_custom = 0.01    # Rate di termalizzazione qubit 3
T1_custom = 1.5     # Temperatura iniziale del qubit 1 (Tc)
T2_custom = 2.0     # Temperatura del qubit 2
T3_custom = 4.0     # Temperatura del qubit 3 (Th)

# Calcolo del punto stazionario con i tuoi parametri
T1_stazionaria, Q1_stazionario = solve_qar_steadystate(
    E1=E1_custom, E2=E2_custom, g=g_custom, 
    p1=p1_custom, p2=p2_custom, p3=p3_custom, 
    T1=T1_custom, T2=T2_custom, T3=T3_custom
)

#ATTENZIONE: stampa a schermo i valori calcolati coi parametri specificati qui sopra, non quelli usati nei grafici
print(f"Temperatura stazionaria T1_S: {T1_stazionaria}")
print(f"Flusso di calore stazionario Q1_S: {Q1_stazionario}")

