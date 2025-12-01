import { createApp } from 'vue'
import './style.css'
import './css/main.css'
import App from './App.vue'
import '@perspecto-cards/components/style/main.css'
import '@perspecto-cards/firebase/setup'
import { createComponent } from '@perspecto-cards/components/main'
import router from "@perspecto-cards/router"

const comp = createComponent()
const app = createApp(App)
app.use(comp)
app.use(router)
app.mount('#app')