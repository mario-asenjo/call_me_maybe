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
- [x] Crear `src/models.py`
- [x] Modelar definición de parámetro
- [x] Modelar definición de función
- [x] Modelar item de prompt
- [x] Modelar resultado final
- [x] Prohibir claves extra en modelos
- [x] Validar tipos soportados (`string`, `number`, `boolean`)
- [x] Validar unicidad de nombres de función

## 3. Errores y carga de entrada
- [x] Crear `src/errors.py`
- [x] Crear `src/input_loader.py`
- [x] Manejar archivo inexistente
- [x] Manejar JSON inválido
- [x] Manejar estructura JSON inválida
- [x] Cargar funciones desde fichero
- [x] Cargar prompts desde fichero
- [x] Resolver rutas por defecto
- [x] Emitir mensajes claros y defendibles

## 4. Integración con `llm_sdk`
- [ ] Crear `src/llm_client.py`
- [ ] Encapsular `Small_LLM_Model`
- [ ] Usar solo API pública del SDK
- [ ] Cargar vocabulario vía `get_path_to_vocab_file()`
- [ ] Exponer `encode`
- [ ] Exponer `decode`
- [ ] Exponer `get_logits_from_input_ids`
- [ ] Tipar y documentar adaptador

## 5. Índices de vocabulario y utilidades de tokenización
- [ ] Crear `src/vocab_index.py`
- [ ] Cargar vocabulario JSON
- [ ] Mapear `token_id -> texto`
- [ ] Normalizar representación de tokens si hace falta
- [ ] Identificar tokens estructurales útiles
- [ ] Diseñar estructura de prefijos para literales válidos
- [ ] Preparar soporte para cadenas
- [ ] Preparar soporte para números
- [ ] Documentar limitaciones de BPE/tokenización

## 6. Diseño de la salida restringida
- [ ] Fijar formato interno restringido:
  - [ ] `{"fn_name":"...","args":{...}}`
- [ ] Fijar formato final serializado:
  - [ ] `{"prompt":"...","fn_name":"...","args":{...}}`
- [ ] Confirmar orden de ensamblaje
- [ ] Confirmar ausencia de claves extra
- [ ] Confirmar que `prompt` se copia del input, no del LLM

## 7. Motor de restricciones
- [ ] Crear `src/constraint_engine.py`
- [ ] Definir estados de la generación
- [ ] Restringir apertura de objeto JSON
- [ ] Restringir clave `fn_name`
- [ ] Restringir valor de `fn_name` al conjunto permitido
- [ ] Restringir transición a `args`
- [ ] Restringir claves de `args` según la función elegida
- [ ] Restringir cierre correcto del objeto
- [ ] Prohibir texto libre
- [ ] Prohibir claves extra
- [ ] Documentar la máquina de estados

## 8. Restricción por tipos
- [ ] Soporte restringido para `string`
- [ ] Soporte restringido para `number`
- [ ] Soporte restringido para `boolean`
- [ ] Validar que los argumentos requeridos estén todos
- [ ] Validar que no aparezcan argumentos no definidos
- [ ] Asegurar que cada paso deja el JSON en estado recuperable

## 9. Selección de función con LLM
- [ ] Diseñar prompt de sistema/instrucción mínimo
- [ ] Incluir definiciones de funciones en contexto
- [ ] Incluir prompt del usuario
- [ ] Confirmar que `fn_name` lo decide el LLM
- [ ] Confirmar que no hay heurísticas por keywords

## 10. Construcción de argumentos con LLM
- [ ] Diseñar extracción restringida de argumentos
- [ ] Confirmar tipado correcto
- [ ] Soportar cadenas vacías
- [ ] Soportar números grandes
- [ ] Soportar caracteres especiales
- [ ] Revisar prompts ambiguos
- [ ] Revisar funciones con múltiples parámetros

## 11. Motor de generación
- [ ] Crear `src/generation_engine.py`
- [ ] Bucle token a token
- [ ] Obtener logits
- [ ] Aplicar máscara de tokens válidos
- [ ] Elegir mejor token válido
- [ ] Actualizar estado
- [ ] Detectar finalización
- [ ] Manejar imposibilidad de continuación válida
- [ ] Imponer límites de longitud razonables

## 12. Ensamblaje y serialización final
- [ ] Crear `src/serializer.py`
- [ ] Parsear salida restringida generada
- [ ] Validarla con Pydantic
- [ ] Añadir `prompt` original
- [ ] Acumular resultados en lista
- [ ] Escribir JSON final bonito y válido
- [ ] Crear carpeta de salida si no existe
- [ ] No subir `output/` al repo

## 13. CLI principal
- [ ] Crear `src/__main__.py`
- [ ] Soportar `--input`
- [ ] Soportar `--output`
- [ ] Definir defaults correctos
- [ ] Propagar errores con mensajes claros
- [ ] Devolver códigos de salida razonables

## 14. Makefile
- [ ] `install`
- [ ] `run`
- [ ] `debug`
- [ ] `clean`
- [ ] `lint`
- [ ] `lint-strict` (opcional)
- [ ] Verificar que los comandos usan `uv`

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

## 17. Peer evaluation readiness
- [ ] Poder explicar por qué esto sí es decodificación restringida
- [ ] Poder explicar por qué no se usan heurísticas
- [ ] Poder explicar el mapeo token ↔ string
- [ ] Poder explicar por qué `prompt` se ensambla al final
- [ ] Poder justificar cada módulo
- [ ] Poder hacer una recodificación pequeña en pocos minutos

## 18. Revisión final antes de entrega
- [ ] `uv sync`
- [ ] `uv run python -m src`
- [ ] `make lint`
- [ ] Tests en verde
- [ ] README completo
- [ ] `output/` no versionado
- [ ] Repo limpio y defendible