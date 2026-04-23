async function sendMessage(){

let input=document.getElementById('user-input');
let msg=input.value;

if(!msg) return;

let chat=document.getElementById('chat-box');

chat.innerHTML+=`
<div class='user'>
<b>You:</b> ${msg}
</div>`;

input.value='';

let r=await fetch('/chat',{
method:'POST',
headers:{
'Content-Type':'application/json'
},
body:JSON.stringify({
message:msg
})
});

let data=await r.json();

chat.innerHTML+=`
<div class='bot'>
<b>Persistent AI:</b> ${data.reply}<br>
<span class='badge'>${data.mode}</span>
</div>`;

chat.scrollTop=chat.scrollHeight;
}
