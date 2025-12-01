<template>
    <router-view v-slot="{ Component }">
        <component :is="Component" :key="route.fullPath"></component>
    </router-view>
</template>
<script setup lang="ts">
import { useRouter, useRoute } from 'vue-router'
const router = useRouter()
const route = useRoute()

const navigateToAdmin = () => {
    router.push('/admin')
}
</script>

<style scoped></style>
