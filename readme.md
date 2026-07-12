# BSL Avatar Prototype

This project develops a prototype of a 3D virtual human that communicates using British Sign Language (BSL). The system includes a JavaScript frontend and a Python backend, with SmolLM integrated to support multi-turn conversations. After a user enters text, the model generates a response, which is matched against the available BSL vocabulary and converted into BSL-style syntax. The corresponding sign motions are then concatenated and interpolated to create a complete animation sequence, which drives the 3D virtual human on the frontend. 

## 1 Environment Setup
Our environment is based on conda
1. Clone the project from git
2. `cd` to the project root
3. Run env_setup.sh as:  
    ```
    chmod +x env_setup.sh
    ./env_setup.sh
    ```

## 2 Program Run
It is recommended to use Chrome as default web browser, since this is a prototype and we have not applied browser adaption. 
1. To run the program, `cd` to the project root
2. Activate the enviroment: `conda activate signavatar`
3. Run: `python app.py`

The first run could be slow, since pretrained models need to be downloaded and the virtual character models need to be cached to frontend.

The first run downloads these pretrained models from Hugging Face:

- `HuggingFaceTB/SmolLM3-3B`
- `urchade/gliner_medium-v2.1`

The download requires an internet connection and sufficient local disk space.
Later runs reuse the locally cached model files.

To stop the application reliably, return to the terminal running `app.py` and
press `Ctrl+C`. Closing the browser tab does not currently guarantee that the
Python server will stop.


## 3 Required Runtime Assets

### Avatar Models

The `models/` directory (under project root) is excluded from Git and must be provided separately.
The frontend expects the directory to contain an avatar manifest, GLB models,
and any snapshot images referenced by the manifest:

```text
models/
|-- avatars.json
|-- woman.glb
|-- man.glb
`-- snapshots/
```

### Sign Motion Data

The backend resolves vocabulary tokens to Three.js motion JSON files. The
current motion root is hard-coded in
`motion_retrieval/motion_sequence_builder.py` as:

```text
/home/ywen/Desktop/smpl_data_threejs/
```

Until this path is made configurable, the motion files must be placed there or
the constant must be updated for the local machine. Each vocabulary token uses
this directory layout:

```text
smpl_data_threejs/
`-- hello/
    `-- hello_0.json
```

The folder and JSON filename are derived from the normalized vocabulary token.
Missing motion files are reported as missing tokens, and a request fails if no
motion JSON can be found for any traced token.


## 4 Project Data Flow
<img src="./static/src/data_flow.png">  

The diagram separates the project into an online interaction pipeline and an
offline data-preparation pipeline. The offline pipeline builds the reusable
motion library that the online pipeline searches when generating an answer.

### Online Interaction Pipeline

1. The user submits text through the frontend. `app.py` exposes the current
   Python HTTP API and serves the browser application. The diagram uses Flask
   as a general backend example, but the current entry point uses
   `http.server` and `socketserver` directly.
2. SmolLM generates the answer. Conversation history can be included when the
   request requires recent or archived context. This behavior is implemented
   in `language_utils/smollm_service.py` and
   `language_utils/chat_history_store.py`.
3. The language pipeline detects entities, normalizes the generated text, and
   prepares BSL-oriented tokens. Entity and text analysis are implemented in
   `language_utils/gliner_service.py` and
   `language_utils/text_analyzer.py`.
4. The vocabulary tree applies token matching against the available BSL
   vocabulary. The main lookup logic is in `language_utils/vocab_tree.py`.
5. `motion_retrieval/motion_sequence_builder.py` resolves matched vocabulary
   tokens to motion clips, concatenates the clips, and inserts transition
   frames to create one continuous Three.js animation.
6. `app.py` returns the generated text, diagnostics, and animation frames to
   the frontend as JSON.
7. The frontend renders the result through three coordinated modules:
   `static/js/avatar.js` loads models and exposes the public avatar API,
   `static/js/avatar_scene.js` owns the Three.js scene and renderer, and
   `static/js/avatar_animation.js` prepares and plays the motion frames.

### Offline Data-Preparation Pipeline

1. BSL vocabulary and source videos are collected from the SignBSL
   dictionary.
2. SMPLest-X extracts SMPL-X body parameters from the source videos. Batch
   inference support is under `batch_infer_smpl/`.
3. Scripts under `preprocess/` convert the SMPL-X axis-angle motion into
   Mixamo-compatible quaternion rotations and Three.js JSON.
4. The processed JSON clips are stored in the external motion library using
   normalized vocabulary-token folders. The online motion-retrieval stage
   reads this library when constructing an avatar response.

The camera panel is currently a preview placeholder. Camera recording,
sign-language recognition, and routing recognized signs into this online
pipeline are planned but are not implemented yet.


## 5 Dataset Collection
We collect the data from https://www.signbsl.com/sign/dictionary.


## 6 Data Preparation
1. To cleanup the raw data collected from online, follow the steps documented in `./vocab_preprocess/readme.md`. This is for cleaning on textual contents only.  

2. To infer smplx parameters, particularly we care about axis-angle here, setup SMPLest-X based on its official repo: https://github.com/MotrixLab/SMPLest-X.  
The batch inference script is under `./batch_infer_smpl/batch_inference.py`.  
For each video, the smplx parameters are stored in a .npz.  

3. We need to convert the axis-angle rotation to quaternion rotations (in .npz) that are compatible with the js frontend (mixamo style) (stored in JSON).  
The batch conversion file is under `./preprocess/batch_axisangle_npz_to_threejs_json.py`

