# Call Me Maybe — Project Checklist

## 0. Base del proyecto
- [x] Crear estructura mínima del repo:
  - [x] `src/`
  - [x] `llm_sdk/`
  - [x] `data/input/`
  - [x] `.github/workflows/`
- [x] Añadir `pyproject.toml`
- [x] Añadir `uv.lock`
- [x] Añadir `Makefile`
- [x] Añadir `.gitignore`
- [ ] Confirmar que `uv sync` funciona desde cero

## 1. Requisitos del subject confirmados
- [x] Python 3.10+
- [x] `flake8` configurado
- [x] `mypy` configurado con los flags requeridos
- [x] Type hints en funciones y variables relevantes
- [x] Docstrings PEP 257
- [x] Manejo robusto de errores
- [x] Clases de dominio con Pydantic
- [x] CLI compatible con:
  - [x] `uv run python -m src`
  - [x] `uv run python -m src --input ... --output ...`
- [x] No usar librerías prohibidas directamente
- [x] No usar métodos/atributos privados de `llm_sdk`

## 2. Modelado y validación de datos
- [x] Crear modelos de dominio
- [x] Modelar definición de parámetro
- [x] Modelar definición de función
- [x] Modelar item de prompt
- [x] Modelar resultado final
- [x] Prohibir claves extra en modelos
- [x] Validar tipos soportados (`string`, `number`, `boolean`)
- [x] Validar unicidad de nombres de función

## 3. Errores y carga de entrada
- [x] Crear errores del dominio
- [x] Crear loader de inputs
- [x] Manejar archivo inexistente
- [x] Manejar JSON inválido
- [x] Manejar estructura JSON inválida
- [x] Cargar funciones desde fichero
- [x] Cargar prompts desde fichero
- [x] Resolver rutas por defecto
- [x] Emitir mensajes claros

## 4. Integración con `llm_sdk`
- [x] Crear adaptador `llm_client`
- [x] Encapsular `Small_LLM_Model`
- [x] Usar solo API pública del SDK
- [x] Cargar vocabulario vía `get_path_to_vocab_file()`
- [x] Exponer `encode`
- [x] Exponer `decode`
- [x] Exponer `get_logits_from_input_ids`
- [x] Tipar y documentar adaptador

## 5. Índices de vocabulario y utilidades de tokenización
- [x] Crear `vocab_loader`
- [x] Cargar vocabulario JSON
- [x] Mapear `token_id -> texto`
- [ ] Normalizar representación de tokens si hace falta
- [ ] Identificar tokens estructurales útiles
- [x] Diseñar estructura de prefijos para literales válidos
- [ ] Preparar soporte para cadenas
- [x] Preparar soporte para números
- [ ] Documentar limitaciones de BPE/tokenización

## 6. Diseño de la salida restringida
- [x] Fijar formato interno restringido:
  - [x] `{"fn_name":"...","args":{...}}`
- [x] Fijar formato final serializado:
  - [x] `{"prompt":"...","fn_name":"...","args":{...}}`
- [x] Confirmar orden de ensamblaje
- [x] Confirmar ausencia de claves extra
- [x] Confirmar que `prompt` se copia del input, no del LLM

## 7. Motor de restricciones
- [x] Crear `constraint_engine`
- [x] Definir estados de la generación
- [x] Restringir apertura de objeto JSON
- [x] Restringir clave `fn_name`
- [x] Restringir valor de `fn_name` al conjunto permitido
- [x] Restringir transición a `args`
- [x] Restringir claves de `args` según la función elegida
- [x] Restringir cierre correcto del objeto
- [x] Prohibir texto libre en las fases implementadas
- [x] Prohibir claves extra en las fases implementadas
- [ ] Documentar la máquina de estados

## 8. Restricción por tipos
- [ ] Soporte restringido para `string`
- [x] Soporte restringido para `number`
- [ ] Soporte restringido para `boolean`
- [ ] Validar que los argumentos requeridos estén todos
- [x] Validar que no aparezcan argumentos no definidos en las fases implementadas
- [ ] Asegurar que cada paso deja el JSON en estado recuperable

## 9. Selección de función con LLM
- [x] Diseñar prompt de sistema/instrucción mínimo
- [x] Incluir definiciones de funciones en contexto
- [x] Incluir prompt del usuario
- [x] Confirmar que `fn_name` lo decide el LLM
- [x] Confirmar que no hay heurísticas por keywords
- [ ] Afinar la calidad semántica de selección para todos los prompts

## 10. Construcción de argumentos con LLM
- [x] Diseñar extracción restringida de argumentos numéricos
- [x] Confirmar tipado correcto para `number`
- [ ] Soportar cadenas vacías
- [x] Soportar números grandes
- [ ] Soportar caracteres especiales
- [ ] Revisar prompts ambiguos
- [x] Revisar funciones con múltiples parámetros numéricos

## 11. Motor de generación
- [x] Crear `generation_engine.py`
- [x] Bucle token a token
- [x] Obtener logits
- [x] Aplicar máscara de tokens válidos
- [x] Elegir mejor token válido
- [x] Actualizar estado
- [x] Detectar finalización
- [x] Manejar imposibilidad de continuación válida
- [x] Imponer límites de longitud razonables

## 12. Ensamblaje y serialización final
- [ ] Crear `serializer.py`
- [ ] Parsear salida restringida generada
- [ ] Validarla con Pydantic
- [ ] Añadir `prompt` original
- [ ] Acumular resultados en lista
- [ ] Escribir JSON final bonito y válido
- [ ] Crear carpeta de salida si no existe
- [x] No subir `output/` al repo

## 13. CLI principal
- [x] Crear `__main__.py`
- [x] Soportar `--input`
- [x] Soportar `--output`
- [x] Definir defaults correctos
- [x] Propagar errores con mensajes claros
- [x] Devolver códigos de salida razonables
- [ ] Sustituir el test de generación única por ejecución real sobre todos los prompts

## 14. Makefile
- [x] `install`
- [x] `run`
- [x] `debug`
- [x] `clean`
- [x] `lint`
- [ ] `lint-strict` (opcional)
- [x] Verificar que los comandos usan `uv`

## 15. Tests
- [ ] Tests de modelos Pydantic
- [ ] Tests de carga de JSON
- [ ] Tests de errores de entrada
- [ ] Tests de `llm_client`
- [ ] Tests de índices de vocabulario
- [ ] Tests de estados del `constraint_engine`
- [ ] Tests de generación acotada
- [ ] Tests end-to-end con input real
- [ ] Casos límite
- [ ] JSON final siempre parseable

## 16. README
- [ ] Primera línea obligatoria de 42
- [ ] Descripción
- [ ] Instrucciones
- [ ] Recursos
- [ ] Uso de IA
- [ ] Explicación del algoritmo
- [ ] Decisiones de diseño
- [ ] Análisis de rendimiento
- [ ] Retos encontrados
- [ ] Estrategia de pruebas
- [ ] Ejemplos de uso

## 17. Bonus readiness
- [x] Base preparada para múltiples modelos LLM
- [x] Base preparada para mecanismos avanzados de recuperación de errores
- [x] Base preparada para trazas / visualización
- [ ] Implementar visualización del proceso de generación
- [ ] Suite comprensiva de tests
- [ ] Reforzar recuperación avanzada de errores
- [ ] Añadir optimizaciones / caches si hacen falta

## 18. Peer evaluation readiness
- [ ] Poder explicar por qué esto sí es decodificación restringida
- [ ] Poder explicar por qué no se usan heurísticas
- [ ] Poder explicar el mapeo token ↔ string
- [ ] Poder explicar por qué `prompt` se ensambla al final
- [ ] Poder justificar cada módulo
- [ ] Poder hacer una recodificación pequeña en pocos minutos

## 19. Revisión final antes de entrega
- [ ] `uv sync`
- [ ] `uv run python -m src`
- [ ] `make lint`
- [ ] Tests en verde
- [ ] README completo
- [x] `output/` no versionado
- [ ] Repo limpio y defendible