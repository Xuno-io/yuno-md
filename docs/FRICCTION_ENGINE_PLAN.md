# Plan de Implementación: FrictionEngine — Cableando la Psicosis en la Arquitectura

## Contexto del Debate

### El Problema Original

El código base de yuno-md tiene una arquitectura sólida (memoria semántica categorizada, destilación por saturación, ventana móvil + hilos), pero su **propuesta de valor principal** —el carácter desafiante, el rigor, la ausencia de validación— residía **exclusivamente en los prompts**.

**La vulnerabilidad:**
- Un prompt es una **sugerencia probabilística**, no una ley
- El LLM puede ignorar el tono desafiante, validar en lugar de cuestionar, o destilar sin los 4 movimientos
- Si la "psicosis" depende del estado de ánimo estocástico del modelo, el sistema es **frágil**

### El Principio Rector

> "Separación de mundos: el determinista (Python) gobierna al probabilístico (LLM)"

La identidad de Yuno debe estar **cableada en la arquitectura**, no pintada en el prompt.

---

## La Solución: FrictionEngine + FrictionFallbackGenerator

### Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────────────┐
│                    NeibotService.get_response()                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
        ┌──────────────────────┴──────────────────────┐
        │ 1. FrictionEngine calcula required_friction │
        │    (determinista, basado en patrones)       │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
        ┌──────────────────────┴──────────────────────────┐
        │ 2. SystemPromptInjector inyecta prompt con      │
        │    required_friction, fatigue_score, etc.       │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌──────────────────────┴──────────────────────┐
        │ 3. LLM genera respuesta (JSON Schema forzado)│
        │    - friction_level (0-3)                   │
        │    - core_argument                          │
        │    - weak_point_exposed                     │
        │    - line_of_flight                         │
        │    - action_vector                          │
        │    - raw_response                           │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
        ┌──────────────────────┴──────────────────────┐
        │ 4. Parsear JSON → ResponseSchema (validación)│
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
        ┌──────────────────────┴──────────────────────┐
        │ 5. FrictionEngine.analyze_and_enforce()     │
        │    - friction_level >= required_friction?   │
        │    - weak_point_exposed existe?             │
        └──────────────────────┬──────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
        ┌───────────────────┐   ┌───────────────────┐
        │ 7a. PASÓ          │   │ 7b. FALLÓ         │
        │ _render_response()│   │ FrictionFallback  │
        │ → texto final     │   │ Generator         │
        └───────────────────┘   │ (sin retry)       │
                                │ → fricción        │
                                │   sintética en    │
                                │   <50ms           │
                                └───────────────────┘
```

### Componentes Clave

#### 1. `FrictionEngine` (Motor de Validación Determinista)

```python
class FrictionLevel(Enum):
    VALIDACION = 0      # Nunca (solo saludos)
    ANALISIS = 1        # Raro
    DESAFIO = 2         # Default
    ANIQUILACION = 3    # Usuario se repite

class ResponseSchema(TypedDict):
    friction_level: FrictionLevel
    friction_justification: str
    core_argument: str
    weak_point_exposed: str | None
    line_of_flight: str | None
    action_vector: str | None
    raw_response: str

class FrictionEngine:
    def analyze_and_enforce(
        self,
        user_message: str,
        history: list[MessagePayload],
        llm_analysis: ResponseSchema
    ) -> tuple[ResponseSchema, bool]:
        """
        Deterministically validate and possibly reject LLM output.
        
        Returns: (validated_schema, was_rejected)
        """
        # 1. Detectar patrones del usuario (determinista)
        user_is_repeating = self._detect_repetition(user_message, history)
        user_is_vague = self._is_vague(user_message)
        
        # 2. Calcular fricción mínima requerida
        required_min_friction = self._calculate_required_friction(
            is_repeating=user_is_repeating,
            is_vague=user_is_vague
        )
        
        # 3. Validar cumplimiento del LLM
        if llm_analysis["friction_level"] < required_min_friction:
            llm_analysis["friction_level"] = required_min_friction
        
        # 4. Forzar requisitos estructurales
        if llm_analysis["friction_level"] >= FrictionLevel.DESAFIO:
            if not llm_analysis.get("weak_point_exposed"):
                raise FrictionViolationError(
                    "Desafío sin punto débil expuesto. Rechazado."
                )
        
        return llm_analysis, False
```

#### 2. `FrictionFallbackGenerator` (Fricción Sintética en Python)

```python
class FrictionFallbackGenerator:
    """
    Cuando el LLM falla en generar fricción, Python la genera.
    Cero llamadas al LLM. Milisegundos.
    """
    
    FALLBACK_TEMPLATES = {
        "vaguedad": [
            "Tu argumento es nebuloso. '{core}'. ¿Qué mecanismo específico "
            "garantiza que eso funcione? No quiero especulación, quiero "
            "causalidad demostrable.",
        ],
        "repetición": [
            "Esto es la tercera vez que mencionas '{core}'. "
            "O no has avanzado o no estás viendo el obstáculo. "
            "El obstáculo es: {weak_point}. Resuélvelo o evoluciona.",
        ],
        "falta_evidencia": [
            "Afirmas '{core}' sin datos. ¿Qué métrica, caso de uso o "
            "evidencia empírica respalda eso? Si no existe, es una suposición.",
        ],
    }
    
    def generate_fallback(
        self,
        user_message: str,
        history: list[MessagePayload],
        llm_partial: dict,
        violation_reason: str
    ) -> ResponseSchema:
        """
        Genera fricción determinista SIN llamar al LLM.
        """
        pattern = self._classify_violation(user_message, history, violation_reason)
        core = llm_partial.get("core_argument", user_message[:100])
        weak_point = self._generate_weak_point_deterministic(user_message, pattern)
        template = self.FALLBACK_TEMPLATES.get(pattern, self.FALLBACK_TEMPLATES["vaguedad"][0])
        
        raw_response = template.format(core=core, weak_point=weak_point)
        
        return ResponseSchema(
            friction_level=FrictionLevel.DESAFIO,
            friction_justification=f"Fallback determinista: {pattern}",
            core_argument=core,
            weak_point_exposed=weak_point,
            line_of_flight=None,
            action_vector=None,
            raw_response=raw_response
        )
```

#### 3. `SystemPromptInjector` (Inyección Dinámica de Contexto)

```python
class SystemPromptInjector:
    def inject(
        self,
        base_prompt: str,
        user_id: str,
        history: list[MessagePayload],
        required_friction: FrictionLevel,
        user_memories: list[dict]
    ) -> str:
        """
        Inyecta contexto determinista en el prompt.
        """
        fatigue_score = self._calculate_user_fatigue(history)
        active_constraints = [
            m["text"] for m in user_memories 
            if m.get("category") == "USER_CONSTRAINTS"
        ]
        
        return f"""{base_prompt}

[CONTEXTO DINAMICO - INYECTADO POR PYTHON]
- User ID: {user_id}
- Turnos en sesión: {len(history)}
- Fatiga detectada: {fatigue_score:.2f}
- Fricción mínima requerida: {required_friction.value}
- Constraints activos: {active_constraints or "Ninguno"}

[INSTRUCCION CRITICA]
NO puedes usar un friction_level menor que {required_friction.value}.
Si lo haces, el motor de fricción rechazará tu respuesta.
"""
```

---

## Estructura de Archivos Propuesta

```
app/services/FrictionService/
├── __init__.py
├── friction_engine.py           # Motor de validación determinista
├── friction_fallback.py          # Generador de fricción sintética
├── friction_injector.py          # Inyección dinámica de prompts
├── friction_schema.py            # TypedDict + Enums
└── friction_service_interface.py # Interface para testability
```

### Archivos Existentes a Modificar

| Archivo | Cambios |
|---------|---------|
| `app/services/NeibotService/neibot_service.py` | Inyectar `FrictionEngine`, `FrictionFallbackGenerator`, `SystemPromptInjector`. Modificar `get_response()` para parsear JSON, validar fricción, aplicar fallback. |
| `app/dependencies/components.py` | Añadir factory functions para crear los nuevos servicios |
| `app/bootstrap/components.py` | Registrar nuevas dependencias en el DI container |

---

## Diferencia Crítica: Antes vs Después

| Antes (Probabilístico) | Después (Determinista) |
|------------------------|------------------------|
| Prompt sugiere "sé desafiante" | Python **exige** `friction_level >= 2` |
| LLM decide si exponer debilidades | Schema JSON **requiere** `weak_point_exposed` |
| Destilación es un prompt | Destilación es un **protocolo con validación** |
| Memoria guarda "lo que el LLM decida" | Prompt **filtra** + Python **valida** categorías |
| LLM falla → **retry** (2-5s) | LLM falla → **fallback determinista** (<50ms) |

---

## Preguntas Pendientes (Antes de Implementar)

### 1. ¿Dónde vive el prompt base de Yuno actualmente?

- ¿En `configuration/` como JSON?
- ¿En un archivo `.prompt`?
- ¿Hardcoded en `neibot_service.py`?

**Necesito saber esto para decidir si el `SystemPromptInjector` lee de archivo o inyecta sobre un string base.**

### 2. ¿Quieres que el JSON schema sea estricto o flexible?

- **Estricto**: Si el LLM devuelve JSON malformado, fallback inmediato (más rápido, menos tolerante)
- **Flexible**: Intentar parsear con `json_repair` o similar antes de fallback (más lento, más tolerante)

### 3. ¿Los templates del `FrictionFallbackGenerator` deben estar en:

- **Archivo externo** (para editar sin recompilar, más flexible)
- **Hardcoded en Python** (más rápido, menos flexible)

### 4. ¿Quieres que el `FrictionEngine` guarde métricas de fallback rate?

- ¿Cuántas veces Python tuvo que generar fricción?
- ¿Qué patrones de fallo son más comunes?
- ¿Para debugging y tuning?

---

## Criterios de Éxito

1. **Latencia garantizada**: Cero retry loops. Fallback en <50ms.
2. **Fricción garantizada**: Si el LLM se ablanda, Python genera fricción.
3. **Costo controlado**: Cero tokens adicionales por fallback.
4. **Degradación elegante**: El fallback usa el `core_argument` del LLM para personalizar la respuesta.
5. **Testability**: Interfaces claras para mockear `FrictionEngine` en tests.

---

## Próximos Pasos

1. Responder las 4 preguntas pendientes
2. Implementar los nuevos archivos en `app/services/FrictionService/`
3. Modificar `NeibotService` para integrar el flujo
4. Añadir factory functions en `app/dependencies/components.py`
5. Registrar dependencias en `app/bootstrap/components.py`
6. Escribir tests unitarios para `FrictionEngine` y `FrictionFallbackGenerator`
7. E2E test: Verificar que la fricción se mantiene incluso cuando el LLM falla

---

**Estado:** Plan validado, pendiente de aprobación para implementación.

**Última actualización:** 2026-03-15
