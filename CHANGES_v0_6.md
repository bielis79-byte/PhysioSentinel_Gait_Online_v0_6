# PhysioSentinel Gait Online v0.6

## Novedades

- Mantiene el flujo de privacidad: vídeo -> `/tmp` -> Pose2Sim/RTMPose -> métricas -> Supabase -> eliminación del vídeo.
- Añade registro de **ayuda técnica**: sin ayuda, bastón, 1/2 muletas, caminador, rollator u otra.
- Guarda la ayuda técnica en `gait_sessions` y permite comparar longitudinalmente solo sesiones realizadas bajo la misma condición.
- Añade control regional de visibilidad: tren inferior, pie/tobillo y tren superior. Esto permite detectar cuándo una ayuda técnica está ocultando puntos relevantes.
- En marcha con ayuda técnica, las métricas del tren superior pueden quedar marcadas como **condicionadas por ayuda técnica**.

## Biomecánica frontal/posterior 2D proyectada

La v0.6 evita etiquetar como flexión/extensión los ángulos calculados desde una vista frontal. En esa vista añade:

- desviación frontal proyectada cadera-rodilla-tobillo D/I;
- orientación distal proyectada del pie D/I;
- inclinación proyectada tobillo-talón (retropié) D/I;
- oblicuidad pélvica proyectada;
- anchura de base relativa tobillos/pelvis;
- diferencias D/I y evolución longitudinal.

## Límites clínicos explícitos

- La **rotación interna/externa real de cadera** es 3D y no se cuantifica directamente con una sola cámara frontal.
- La orientación del pie se usa como **proxy distal**, pero no se atribuye automáticamente a la cadera.
- La **pronación** también es 3D. La inclinación tobillo-talón puede sugerir cambios de inversión/eversión proyectada, pero no equivale por sí sola a pronación clínica.
- Los valores siguen siendo dependientes de perspectiva; las medidas métricas requerirán calibración/homografía o 3D.

## Supabase

Ejecutar de nuevo `SUPABASE_SETUP.sql`. Es compatible con v0.5 y añade con `IF NOT EXISTS`:

- `assistive_device`
- `assisted_gait`
- `frontal_orientation`

No elimina datos existentes.
