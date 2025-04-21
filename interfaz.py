import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import os

# Lista de clases
clasesArañas = ["Violinista", "Lobo", "Saltarina", "DeJardin", "ViudaNegra"]

# informacion de las arañas
info_aranas = {
    "Violinista": """
 Nombre común: Araña Violinista
Nombre científico: Loxosceles reclusa
Peligrosidad: Sí, es venenosa y potencialmente peligrosa para humanos.

 Síntomas de la picadura:
- Dolor leve al inicio que puede aumentar
- Enrojecimiento e hinchazón
- Formación de una úlcera o necrosis en la piel
- Fiebre, escalofríos, náuseas en casos severos

 Qué hacer:
- Lavar la zona con agua y jabón
- No aplicar calor (puede activar el veneno)
- Aplicar compresas frías
- Acudir inmediatamente al médico
""",

    "Lobo": """
Nombre común: Araña Lobo
Nombre científico: Lycosa spp.
Peligrosidad: No, su picadura no es peligrosa para humanos.

Síntomas de la picadura:
- Dolor local leve o moderado
- Enrojecimiento y leve inflamación

Qué hacer:
- Lavar la zona afectada
- Aplicar hielo o compresas frías
- Consultar a un médico si hay una reacción alérgica
""",

    "Saltarina": """
Nombre común: Araña Saltarina
Nombre científico: Salticidae (familia)
Peligrosidad: No es peligrosa para humanos, muy inofensiva.

Síntomas de la picadura:
- Raramente muerde
- Si muerde, provoca irritación leve

Qué hacer:
- Lavar con agua y jabón
- No se requiere tratamiento médico a menos que haya alergia
""",

    "DeJardin": """
Nombre común: Araña de Jardín
Nombre científico: Araneus diadematus
Peligrosidad: No, es completamente inofensiva.

Síntomas de la picadura:
- No suele morder
- En caso raro, picadura leve parecida a la de un mosquito

Qué hacer:
- No se requiere atención médica
- Solo limpiar la zona si hay molestia
""",

    "ViudaNegra": """
Nombre común: Viuda Negra
Nombre científico: Latrodectus mactans
Peligrosidad: Sí, es muy venenosa y peligrosa para humanos.

Síntomas de la picadura:
- Dolor intenso en la zona
- Calambres musculares
- Náuseas, vómitos, fiebre
- Hipertensión, dificultad respiratoria en casos severos

Qué hacer:
- Buscar atención médica urgente
- No aplicar torniquetes
- Mantener a la persona calmada y en reposo
- Si es posible, capturar la araña para identificación
"""
}

# Procesamiento de imagen a subir
def preprocesarImagen(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen = cv2.resize(imagen, (64, 64))
    imagen = imagen.astype('float32') / 255.0
    return imagen
#interfaz
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Arañas Peligrosas")
        self.root.geometry("1280x720")
        self.root.resizable(False, False)

        self.modelo = tf.keras.models.load_model("modeloSpyder.h5")

        self.title_label = tk.Label(root, text="DETECCIÓN DE ARAÑAS", font=("Helvetica", 24, "bold"))
        self.title_label.pack(pady=20)

        self.main_frame = tk.Frame(root, padx=20, pady=20)
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        self.image_frame = tk.Frame(self.main_frame, bd=2, relief="solid", width=700, height=357)
        self.image_frame.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky="nsew")
        self.image_frame.grid_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text="IMAGEN\nSUBIDA",
            font=("Helvetica", 14, "bold"),
            justify="center"
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        self.select_btn = tk.Button(
            self.main_frame,
            text="Seleccionar Imagen",
            command=self.seleccionar_imagen,
            width=30,
            height=3,
            bg="#007acc",
            fg="white",
            activebackground="#005a99"
        )
        self.select_btn.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.info_btn = tk.Button(
            self.main_frame,
            text="Obtener información",
            command=self.abrirInformacion,
            width=30,
            height=3,
            bg="#14b814",
            fg="white",
            activebackground="#005a99"
        )
        self.info_btn.grid(row=1, column=1, padx=10, pady=10, sticky="n")

        self.salir_btn = tk.Button(
            self.main_frame,
            text="SALIR",
            command=self.root.quit,
            width=25,
            height=3,
            bg="red",
            fg="white",
            activebackground="#cc0000",
            activeforeground="white"
        )
        self.salir_btn.grid(row=2, column=1, sticky="se", padx=10, pady=10)

        self.imagen = None
        self.ruta_imagen = None
        self.resultado_prediccion = None

    def seleccionar_imagen(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if archivo:
            self.ruta_imagen = archivo
            imagen = Image.open(archivo).resize((600, 315)).convert("RGB")
            self.imagen = ImageTk.PhotoImage(imagen)
            self.image_label.configure(image=self.imagen, text="")

            # clasificacion de la imagen seleccionada
            self.resultado_prediccion = self.clasificar_imagen(archivo)

    def clasificar_imagen(self, ruta_imagen):
        try:
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                raise ValueError("Imagen no válida")

            imagenProcesada = preprocesarImagen(imagen)
            imagenProcesadaExpandida = np.expand_dims(imagenProcesada, axis=0)

            prediccion = self.modelo.predict(imagenProcesadaExpandida)
            clasePredicha = np.argmax(prediccion)
            confianza = np.max(prediccion)

            clase = clasesArañas[clasePredicha]
            print(f"Clase predicha: {clase} - Confianza: {confianza:.2%}")
            return clase
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo clasificar la imagen.\n{str(e)}")
            return None

    def abrirInformacion(self):
        if not self.resultado_prediccion:
            messagebox.showwarning("Advertencia", "Primero selecciona una imagen para analizar.")
            return

        info_texto = info_aranas.get(self.resultado_prediccion, "No hay información disponible.")

        info_ventana = tk.Toplevel(self.root)
        info_ventana.title("Información de la Araña")
        info_ventana.geometry("600x400")
        info_ventana.resizable(False, False)

        titulo_info = tk.Label(info_ventana, text="INFORMACIÓN DETECTADA", font=("Helvetica", 16, "bold"))
        titulo_info.pack(pady=10)

        texto_info = tk.Text(info_ventana, wrap=tk.WORD, font=("Arial", 12), padx=10, pady=10)
        texto_info.pack(fill="both", expand=True, padx=20, pady=10)
        texto_info.insert(tk.END, info_texto)
        texto_info.config(state="disabled")

        cerrar_btn = tk.Button(info_ventana, text="Cerrar", command=info_ventana.destroy, bg="red", fg="white", width=15)
        cerrar_btn.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
