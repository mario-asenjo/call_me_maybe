## Checklist de reconstrucción de `call-me-maybe`

### Fase 0 — Limpieza y decisiones base
- [ ] Eliminar imports irrelevantes o accidentales del código de `src/`
- [ ] Congelar formato interno: `{"fn_name":"...","args":{...}}`
- [ ] Congelar formato final: `{"prompt":"...","fn_name":"...","args":{...}}`
- [ ] Mantener solo API pública de `llm_sdk`
- [ ] Soportar tipos: `string`, `number`, `integer`, `boolean`
- [ ] Mantener el orden de parámetros del schema en la salida

### Fase 1 — Base de dominio e inputs
- [ ] Validar definiciones de funciones con Pydantic
- [ ] Validar prompts con Pydantic
- [ ] Detectar funciones duplicadas
- [ ] Propagar errores de lectura/JSON con mensajes claros

### Fase 2 — Motor estructural nuevo
- [ ] Reescribir `ConstraintState` para separar estructura y semántica
- [ ] Restringir selección de función al conjunto permitido
- [ ] Restringir keys de `args` al schema de la función elegida
- [ ] Restringir cierres `,`, `}` y `}}` según estado
- [ ] Restringir cada valor a continuaciones exactas de candidatos completos

### Fase 3 — Constructor de candidatos
- [ ] Crear `engine/value_candidates.py`
- [ ] Extraer números en orden de aparición
- [ ] Serializar `number` siempre como float JSON
- [ ] Serializar `integer` siempre como entero JSON
- [ ] Extraer booleanos plausibles
- [ ] Extraer strings quoted exactos
- [ ] Extraer rutas, encodings, templates y SQL quoted
- [ ] Construir candidatos regex genéricos a partir del lenguaje del prompt
- [ ] Mantener fallback unquoted limitado y ordenado
- [ ] Deduplicar sin romper reutilización legítima

### Fase 4 — Generation engine limpio
- [ ] Construir contexto compacto para el LLM
- [ ] Incluir catálogo de funciones
- [ ] Incluir prompt de usuario
- [ ] Incluir función ya seleccionada
- [ ] Incluir argumentos ya fijados
- [ ] Incluir candidatos activos del valor actual
- [ ] Cachear tokenizaciones y contextos repetidos

### Fase 5 — Validación y serialización final
- [ ] Validar `fn_name` existente
- [ ] Validar que todos los argumentos requeridos están presentes
- [ ] Prohibir claves extra
- [ ] Normalizar tipos numéricos
- [ ] Añadir `prompt` original
- [ ] Crear directorio de salida si no existe
- [ ] Escribir JSON final bonito y parseable

### Fase 6 — CLI final
- [ ] Cargar definiciones y prompts
- [ ] Generar todos los resultados
- [ ] Validar cada resultado antes de añadirlo
- [ ] Escribir `data/output/function_calling_results.json`
- [ ] Terminar con códigos de salida claros

### Fase 7 — Validación técnica
- [ ] `uv sync`
- [ ] `uv run python -m src`
- [ ] `flake8`
- [ ] `mypy`
- [ ] Pruebas end-to-end del set público
- [ ] Pruebas con casos estilo privado: integer, SQL, path, template
- [ ] Medir batch total y dejarlo por debajo del objetivo

### Fase 8 — Entrega
- [ ] README completo
- [ ] Explicación del algoritmo de decodificación restringida
- [ ] Explicación de decisiones de diseño
- [ ] Estrategia de pruebas y rendimiento
