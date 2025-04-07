import { useState } from 'react'
import { Box, Container, Typography, Button, Paper, Grid } from '@mui/material'
import { uploadImage } from './services/api'

function App() {
  const [originalImage, setOriginalImage] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>, type: 'image' | 'mask') => {
    const file = event.target.files?.[0]
    if (!file) return

    setLoading(true)
    try {
      const imageUrl = URL.createObjectURL(file)
      setOriginalImage(imageUrl)

      const processedUrl = await uploadImage(file, type)
      setProcessedImage(processedUrl)
    } catch (error) {
      console.error('Error processing image:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          图像抠图演示
        </Typography>

        <Box sx={{ display: 'flex', gap: 3 }}>
          <Box sx={{ flex: 1 }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                原始图像
              </Typography>
              {originalImage && (
                <img
                  src={originalImage}
                  alt="Original"
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              )}
            </Paper>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                处理结果
              </Typography>
              {processedImage && (
                <img
                  src={processedImage}
                  alt="Processed"
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              )}
            </Paper>
          </Box>
        </Box>

        <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            component="label"
            disabled={loading}
          >
            上传图片进行抠图
            <input
              type="file"
              hidden
              accept="image/*"
              onChange={(e) => handleFileUpload(e, 'image')}
            />
          </Button>

          <Button
            variant="contained"
            component="label"
            disabled={loading}
          >
            上传图片生成掩码
            <input
              type="file"
              hidden
              accept="image/*"
              onChange={(e) => handleFileUpload(e, 'mask')}
            />
          </Button>
        </Box>
      </Box>
    </Container>
  )
}

export default App
