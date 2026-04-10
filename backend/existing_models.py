import numpy as np
from PIL import Image

try:
    import pywt
    DWT_AVAILABLE = True
except ImportError:
    DWT_AVAILABLE = False

class MSBModel:
    @staticmethod
    def embed(cover_arr, payload_bytes):
        stego = cover_arr.copy()
        bits = []
        for b in payload_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        h, w, c = cover_arr.shape
        idx = 0
        for i in range(h):
            for j in range(w):
                for ch in range(c):
                    if idx < len(bits):
                        val = stego[i,j,ch]
                        val = (val & 0x7F) | (bits[idx] << 7)
                        stego[i,j,ch] = val
                        idx += 1
                    else:
                        return stego
        return stego

    @staticmethod
    def extract(stego_arr, length_bytes):
        h, w, c = stego_arr.shape
        bits = []
        for i in range(h):
            for j in range(w):
                for ch in range(c):
                    bits.append((stego_arr[i,j,ch] >> 7) & 1)
        data = bytearray()
        for k in range(0, min(len(bits), length_bytes*8), 8):
            if k+8 > len(bits): break
            byte = 0
            for b in bits[k:k+8]:
                byte = (byte << 1) | b
            data.append(byte)
        return bytes(data)

class DCTJStegModel:
    @staticmethod
    def embed(cover_arr, payload_bytes):
        from scipy.fftpack import dct, idct
        stego = cover_arr.copy()
        ycbcr = np.array(Image.fromarray(cover_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        h, w = y.shape
        bits = []
        for b in payload_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        bit_idx = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = y[i:i+8, j:j+8].copy()
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_int = np.round(dct_block).astype(np.int32)
                for u in range(1, 8):
                    for v in range(1, 8):
                        if bit_idx >= len(bits):
                            break
                        if 4 <= u+v <= 10:
                            dct_int[u, v] = (dct_int[u, v] & 0xFE) | bits[bit_idx]
                            bit_idx += 1
                    if bit_idx >= len(bits): break
                dct_block = dct_int.astype(np.float32)
                block_new = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                y[i:i+8, j:j+8] = np.clip(block_new, 0, 255)
        ycbcr[:,:,0] = y
        return np.array(Image.fromarray(ycbcr, 'YCbCr').convert('RGB'))

    @staticmethod
    def extract(stego_arr, length_bytes):
        from scipy.fftpack import dct, idct
        ycbcr = np.array(Image.fromarray(stego_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        h, w = y.shape
        bits = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = y[i:i+8, j:j+8]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_int = np.round(dct_block).astype(np.int32)
                for u in range(1, 8):
                    for v in range(1, 8):
                        if len(bits) >= length_bytes*8:
                            break
                        if 4 <= u+v <= 10:
                            bits.append(dct_int[u, v] & 1)
                    if len(bits) >= length_bytes*8: break
                if len(bits) >= length_bytes*8: break
            if len(bits) >= length_bytes*8: break
        data = bytearray()
        for k in range(0, len(bits), 8):
            if k+8 > len(bits): break
            byte = 0
            for b in bits[k:k+8]:
                byte = (byte << 1) | b
            data.append(byte)
        return bytes(data)

class DWTHaarModel:
    @staticmethod
    def embed(cover_arr, payload_bytes):
        if not DWT_AVAILABLE:
            return cover_arr
        stego = cover_arr.copy()
        ycbcr = np.array(Image.fromarray(cover_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        coeffs = pywt.dwt2(y, 'haar')
        LL, (LH, HL, HH) = coeffs
        bits = []
        for b in payload_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        LL_int = np.round(LL).astype(np.int32)
        flat = LL_int.flatten()
        for i in range(min(len(bits), len(flat))):
            flat[i] = (flat[i] & 0xFE) | bits[i]
        LL_new = flat.reshape(LL.shape).astype(np.float32)
        y_new = pywt.idwt2((LL_new, (LH, HL, HH)), 'haar')
        y_new = np.clip(y_new, 0, 255).astype(np.uint8)
        ycbcr[:,:,0] = y_new
        return np.array(Image.fromarray(ycbcr, 'YCbCr').convert('RGB'))

    @staticmethod
    def extract(stego_arr, length_bytes):
        if not DWT_AVAILABLE:
            return b''
        ycbcr = np.array(Image.fromarray(stego_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        coeffs = pywt.dwt2(y, 'haar')
        LL, _ = coeffs
        LL_int = np.round(LL).astype(np.int32)
        bits = []
        for val in LL_int.flatten():
            bits.append(val & 1)
        data = bytearray()
        for k in range(0, min(len(bits), length_bytes*8), 8):
            if k+8 > len(bits): break
            byte = 0
            for b in bits[k:k+8]:
                byte = (byte << 1) | b
            data.append(byte)
        return bytes(data)