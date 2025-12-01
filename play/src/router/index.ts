import {createRouter, createWebHistory, Router} from 'vue-router'; 

const routes = [
    {path: '/', component: () => import('@/views/dashboard.vue')},
    {path: '/admin', component: () => import('@/views/admin.vue'), name: "Admin"},
    {path: '/analytics', component: () => import('../views/admin.vue'), name: "Analytics"}, 
    {path: '/message', component: () => import('../views/admin.vue'), name: "Message Board"},
    {path: '/papers', component: () => import('../views/admin.vue'), name: "Manage Papers"},    
    {path: '/account', component: () => import('../views/admin.vue'), name: "My Account",},
]

const router: Router = createRouter({
    history: createWebHistory(),
    routes
})


export default router; 