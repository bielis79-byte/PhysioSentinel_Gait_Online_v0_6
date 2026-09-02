# PhysioSentinel Gait Online v0.6

Versión online con Pose2Sim + RTMPose/HALPE26, Supabase e histórico longitudinal.

## Flujo de privacidad

El vídeo se guarda únicamente en el almacenamiento temporal del servidor Streamlit, se procesa, se calculan las métricas y después se elimina junto con los JSON/archivos de Pose2Sim. Supabase conserva pacientes seudonimizados, sesiones y métricas; no conserva el vídeo.

## Actualización desde v0.5

1. Sustituir en GitHub `streamlit_app.py`, `requirements.txt` y `packages.txt` por los de esta carpeta.
2. Ejecutar `SUPABASE_SETUP.sql` en Supabase SQL Editor. Es una migración no destructiva.
3. Mantener Python **3.11** en Streamlit Community Cloud.
4. Mantener en Streamlit Secrets:

```toml
SUPABASE_URL = "https://TU-PROYECTO.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "TU_SERVICE_ROLE_KEY"
GAIT_APP_PASSWORD = "TU_CONTRASEÑA"
```

## Ayudas técnicas

Antes de crear la sesión se selecciona si la marcha se realiza sin ayuda, con bastón, muletas, caminador o rollator. La app calcula visibilidad por regiones y conserva esa condición en el histórico para evitar comparar de forma automática marchas realizadas bajo condiciones distintas.

## Plano frontal/posterior

La app diferencia ahora la vista frontal de la lateral. La frontal aporta proxies 2D de alineación de rodilla, orientación del pie, inclinación del retropié, oblicuidad pélvica y anchura de base relativa. No se presentan estos proxies como rotación anatómica de cadera ni como pronación 3D.

## Plano lateral

Mantiene las curvas 2D proyectadas de cadera, rodilla, tobillo y hombro, además de cadencia/regularidad/asimetría experimental.
