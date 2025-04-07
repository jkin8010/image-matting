import axios from 'axios';

const API_BASE_URL = 'http://localhost:3080';

export const uploadImage = async (file: File, endpoint: 'image' | 'mask') => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/rembg/${endpoint}`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
    });

    return URL.createObjectURL(response.data);
}; 