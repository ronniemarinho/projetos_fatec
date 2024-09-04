from flask import Flask, send_file
import qrcode

app = Flask(__name__)

@app.route('/')
def generate_qr_code():
    # Dados para o QR Code
    data = "https://www.example.com"  # Substitua pelo link desejado

    # Gerar o QR Code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # Salvar o QR Code como um arquivo temporário
    qr_img_path = "qr_code.png"
    qr_img.save(qr_img_path)

    # Enviar o arquivo para download
    return send_file(qr_img_path, as_attachment=True)

if __name__ == '__main__':
    app.run()
