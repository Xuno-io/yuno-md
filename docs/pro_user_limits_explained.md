# Sistema de Límites: Usuarios Free vs Pro

## Resumen Ejecutivo

El sistema utiliza dos estrategias diferentes para controlar costos según el tipo de usuario:

| Usuario | Límite por Hilo | Mensajes al LLM | Costo por Mensaje |
|---------|-----------------|-----------------|-------------------|
| **Free** | 50 mensajes | Últimos 20 | Variable (debe contar) |
| **Pro** | ∞ Infinito | Últimos 20 | Constante |

---

## ¿Por qué Free tiene límite y Pro no?

### El Problema del Costo

El costo de cada mensaje **no depende de cuántos mensajes lleva el usuario en total**, sino de **cuántos tokens envías al LLM en cada request**.

```
Mensaje #1:   Envías 1 mensaje   → $0.01
Mensaje #50:  Envías 50 mensajes → $0.50  ← ¡50x más caro!
Mensaje #100: Envías 100 mensajes → $1.00 ← Hemorragia de tokens
```

### La Solución: Rolling Window

Aplicamos una **ventana deslizante de 20 mensajes**. Sin importar cuántos mensajes lleve el hilo, siempre enviamos solo los últimos 20 al LLM:

```
Mensaje #1:    Envías 1 mensaje   → $0.01
Mensaje #50:   Envías 20 mensajes → $0.20
Mensaje #5000: Envías 20 mensajes → $0.20  ← Costo plano
```

---

## Lógica por Tipo de Usuario

### Usuario Free (Modo Auditoría)

```python
fetch_limit = max_history_turns  # 50
```

**¿Por qué traemos 50?** Porque necesitamos **contar** cuántos mensajes lleva el hilo para decidir si bloquearlo.

- Si solo traemos 20, `len(history)` siempre sería ≤ 20
- La condición `if len(history) >= 50` nunca se cumpliría
- El usuario Free tendría chat infinito (bug)

**Flujo:**
1. Fetch de 50 mensajes
2. ¿`len(history) >= 50`? → Bloquear con mensaje de límite
3. Si no, aplicar rolling window (últimos 20) y responder

### Usuario Pro (Modo Eficiencia)

```python
fetch_limit = ROLLING_WINDOW_SIZE  # 20
```

**¿Por qué solo 20?** Porque Pro **no tiene límite por hilo**. No necesitas contar nada, solo necesitas los últimos 20 para el contexto del LLM.

**Flujo:**
1. Fetch de 20 mensajes (el mínimo necesario)
2. Enviar al LLM y responder
3. No hay validación de límite

---

## ¿Por qué Pro tiene hilos infinitos?

### Argumento Financiero

Con rolling window, el costo es **constante** sin importar la longitud del hilo:

| Mensaje # | Tokens enviados | Costo |
|-----------|-----------------|-------|
| 1 | ~500 | $0.01 |
| 100 | ~2000 (20 msgs) | $0.04 |
| 5,000 | ~2000 (20 msgs) | $0.04 |

**No hay razón financiera para limitar a un usuario que pagó.**

### Argumento de Experiencia

El usuario Pro pagó por una experiencia **sin barreras**. Interrumpir su conversación en el mensaje 200 cuando el costo del mensaje 5000 es idéntico sería artificialmente restrictivo.

---

## Configuración

```json
{
    "MAX_HISTORY_TURNS": 50,      // Límite para usuarios Free
    "MAX_HISTORY_TURNS_PRO": 20   // Rolling window (no es un límite)
}
```

> **Nota sobre nomenclatura:** `MAX_HISTORY_TURNS_PRO` es técnicamente el tamaño del rolling window, no un "límite". El nombre puede ser confuso pero la funcionalidad es correcta.

---

## Código Relevante

```python
is_pro = self.user_service.is_user_pro(event.sender_id)
max_history_turns = self.user_service.get_user_max_history_turns(event.sender_id)

ROLLING_WINDOW_SIZE = 20

# Pro: solo trae lo necesario para el LLM
# Free: trae hasta el límite para poder contarlo
fetch_limit = ROLLING_WINDOW_SIZE if is_pro else max_history_turns

history = await self.__build_reply_history(event, fetch_limit)

# Solo Free tiene límite por hilo
if not is_pro and len(history) >= max_history_turns:
    await event.reply("Límite alcanzado...")
    return

# Ambos usan rolling window para el contexto del LLM
context_for_dspy = history[-ROLLING_WINDOW_SIZE:]
```

---

## TL;DR

- **Free:** Traemos 50 para contar → Bloqueamos en 50 → Enviamos 20 al LLM
- **Pro:** Traemos 20 → No bloqueamos nunca → Enviamos 20 al LLM
- **Costo Pro constante:** El mensaje #5000 cuesta igual que el #1
- **El código está correcto.** 🚀
