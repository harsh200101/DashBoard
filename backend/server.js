const express = require('express');
const app = express();
const PORT = process.env.PORT || 8080;

app.get('/price-predict', (req, res) => {
   print("hello")
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
